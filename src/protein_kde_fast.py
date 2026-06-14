#!/usr/bin/env python3
"""
Fast KDE-based per-sample sigma using Random Fourier Features (RFF)
over multi-feature antibody descriptors.

- Input CSV (gzip) must have: partition, fv_heavy_aho, fv_light_aho
- Train rows get data-dependent sigma; non-train rows get fallback.
- Works with millions of rows: uses batching for feature → RFF → mean embedding.

Requires:
  pip install pandas numpy scikit-learn biopython joblib tqdm
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Iterable
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import StandardScaler

# -------------------- Config --------------------
INPUT_GZ  = "/data0/srinu_pd/walk-jump/data/poas.csv.gz"
OUTPUT_GZ = "/data0/srinu_pd/walk-jump/data/poas_with_sigma_0.4to0.6_copy.csv.gz"

# Feature + RFF params
FEATURE_COLS = ["gravy_fv", "mw_fv", "pI_fv", "aromaticity_fv", "instability_fv", "sheet_frac_fv"]
D_RFF        = 1024         # number of random features (512–2048 good)
BANDWIDTH    = 0.7          # Gaussian kernel width AFTER z-scoring (tune 0.3–1.2)
SUBSAMPLE_N  = 200_000      # cap to fit scaler / pick bandwidth (random sample of train)
BATCH        = 50_000       # batch size for streaming over big train sets
num_levels = 50
# Sigma mapping (keep your preferred [1.0, 2.0])
SIGMA_MIN, SIGMA_MAX = 0.4,0.6 #1.0, 2.0
P_LO, P_HI = 0.00, 1.0     # robust clipping percentiles
ALPHA = 1.0 # or 0.7               # temper extremes: (1 - z)^ALPHA
FALLBACK_SIGMA = 1.0

SAVE_ARTEFACTS = True
RNG_SEED = 42
# ------------------------------------------------


def clean_seq(s: str) -> str:
    return str(s).replace("-", "")


def fv_features(seq_fv: str) -> Tuple[float, float, float, float, float, float]:
    """Return (gravy, mw, pI, aromaticity, instability, sheet_frac)."""
    try:
        pa = ProteinAnalysis(seq_fv)
        helix, turn, sheet = pa.secondary_structure_fraction()
        return (
            float(pa.gravy()),
            float(pa.molecular_weight()),
            float(pa.isoelectric_point()),
            float(pa.aromaticity()),
            float(pa.instability_index()),
            float(sheet),
        )
    except Exception:
        return (np.nan,)*6


def build_features_for_indices(df: pd.DataFrame, idxs: np.ndarray) -> np.ndarray:
    """Compute feature matrix for the given row indices of df (heavy+light concatenated)."""
    hv = df.loc[idxs, "fv_heavy_aho"].astype(str)
    lv = df.loc[idxs, "fv_light_aho"].astype(str)
    fv = (hv.map(clean_seq) + lv.map(clean_seq)).tolist()
    out = np.empty((len(fv), 6), dtype=np.float64)
    for i, s in enumerate(fv):
        out[i] = fv_features(s)
    return out


def fit_scaler_on_subsample(df_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on a random subsample of train for speed/stability."""
    n = len(df_train)
    rng = np.random.default_rng(RNG_SEED)
    if n > SUBSAMPLE_N:
        idxs = df_train.index.to_numpy()
        choose = rng.choice(idxs, size=SUBSAMPLE_N, replace=False)
    else:
        choose = df_train.index.to_numpy()

    X_sub = build_features_for_indices(df_train, choose)
    mask = np.all(np.isfinite(X_sub), axis=1)
    if mask.sum() < 3:
        raise RuntimeError("Not enough finite features in subsample to fit scaler.")
    scaler = StandardScaler().fit(X_sub[mask])
    return scaler


def init_rff(F: int, d_rff: int, bandwidth: float, rng: np.random.Generator):
    """
    Random Fourier Features for Gaussian kernel:
      phi(x) = sqrt(2/D) * cos(W^T x + b), W_ij ~ N(0, 1/h^2), b_j ~ U[0, 2π]
    """
    W = rng.normal(loc=0.0, scale=1.0/bandwidth, size=(F, d_rff)).astype(np.float64)
    b = rng.uniform(0.0, 2*np.pi, size=(d_rff,)).astype(np.float64)
    scale = math.sqrt(2.0 / d_rff)
    return W, b, scale


def rff_transform(Xs: np.ndarray, W: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    Z = Xs @ W + b  # (N, D_RFF)
    return scale * np.cos(Z)


def stream_train_mu(df_train: pd.DataFrame, scaler: StandardScaler, W, b, scale) -> np.ndarray:
    """Compute the mean embedding μ = E[phi(x)] over TRAIN, in batches."""
    mu = np.zeros(W.shape[1], dtype=np.float64)
    total = 0

    idxs = df_train.index.to_numpy()
    for start in tqdm(range(0, len(idxs), BATCH), desc="RFF μ (train batches)"):
        batch_idx = idxs[start:start+BATCH]
        X = build_features_for_indices(df_train, batch_idx)
        mask = np.all(np.isfinite(X), axis=1)
        if mask.any():
            Xs = scaler.transform(X[mask])
            Phi = rff_transform(Xs, W, b, scale)  # (m, D_RFF)
            mu += Phi.sum(axis=0)
            total += Phi.shape[0]

    if total < 3:
        raise RuntimeError("Too few valid train rows to compute RFF mean embedding.")
    mu /= total
    return mu


def score_train_densities(df_train: pd.DataFrame, scaler, W, b, scale, mu) -> pd.Series:
    """Return density proxy dens(x) ≈ phi(x)·μ for each TRAIN row (NaN if invalid)."""
    idxs = df_train.index.to_numpy()
    dens_all = np.full(len(idxs), np.nan, dtype=np.float64)

    pos = 0
    for start in tqdm(range(0, len(idxs), BATCH), desc="Scoring dens (train)"):
        batch_idx = idxs[start:start+BATCH]
        X = build_features_for_indices(df_train, batch_idx)
        mask = np.all(np.isfinite(X), axis=1)
        if np.any(mask):
            Xs = scaler.transform(X[mask])
            Phi = rff_transform(Xs, W, b, scale)
            dens = Phi @ mu
            dens_all[start:start+BATCH][mask] = dens
        pos += len(batch_idx)

    return pd.Series(dens_all, index=idxs)


def density_to_sigma(dens: np.ndarray,
                     smin=SIGMA_MIN, smax=SIGMA_MAX,
                     lo=P_LO, hi=P_HI, alpha=ALPHA) -> np.ndarray:
    """Robust, tempered inverse-density mapping into [smin, smax]."""
    d = dens.copy()
    # Handle constant/NaN safely
    finite = np.isfinite(d)
    if finite.sum() == 0:
        return np.full_like(d, (smin + smax) / 2.0)
    qlo, qhi = np.quantile(d[finite], [lo, hi])
    den = max(qhi - qlo, 1e-8)
    z = (np.clip(d, qlo, qhi) - qlo) / den  # 0..1
    np.save('/data0/srinu_pd/walk-jump/data/density.npy',z)
    z = (1.0 - z) ** alpha                   # inverse + temper
    return smin + (smax - smin) * z
    # sigma_cont = smin + (smax - smin) * z

    # # Quantize to N discrete σ levels (evenly spaced)
    # sigma_levels = np.linspace(smin, smax, num_levels)
    # sigma_disc = np.array([sigma_levels[np.argmin(np.abs(s - sigma_levels))] for s in sigma_cont])

    # return sigma_disc


def main():
    df = pd.read_csv(INPUT_GZ, compression="gzip")
    assert {"partition", "fv_heavy_aho", "fv_light_aho"}.issubset(df.columns), \
        "Input must have columns: partition, fv_heavy_aho, fv_light_aho"

    df_train = df[df.partition == "train"].copy()

    # 1) Fit scaler on subsample for z-scoring
    scaler = fit_scaler_on_subsample(df_train)

    # 2) Init RFF params
    rng = np.random.default_rng(RNG_SEED)
    F = len(FEATURE_COLS)  # 6
    W, b, scale = init_rff(F, D_RFF, BANDWIDTH, rng)

    # 3) Streaming pass to compute μ over TRAIN
    mu = stream_train_mu(df_train, scaler, W, b, scale)

    # 4) Streaming pass to compute dens per TRAIN row
    dens_series = score_train_densities(df_train, scaler, W, b, scale, mu)

    # 5) Map density → sigma for TRAIN; non-train = fallback
    sigma_train = density_to_sigma(dens_series.values).round(3)
    sigma_full = pd.Series(index=df.index, dtype=float)
    sigma_full.loc[df_train.index] = sigma_train
    sigma_full.loc[df.partition != "train"] = FALLBACK_SIGMA
    df["sigma"] = sigma_full.values

    # 6) Write gzip CSV
    df.to_csv(OUTPUT_GZ, index=False, compression="gzip")
    print(f"[OK] wrote {OUTPUT_GZ}")
    print(f"[Stats] train sigma avg={np.nanmean(sigma_train):.4f} "
          f"min={np.nanmin(sigma_train):.4f} max={np.nanmax(sigma_train):.4f}")

    # 7) Save artefacts (reproducibility)
    if SAVE_ARTEFACTS:
        base = os.path.splitext(OUTPUT_GZ)[0]
        joblib.dump(scaler, base + "_scaler.joblib")
        joblib.dump({"W": W, "b": b, "scale": scale, "bandwidth": BANDWIDTH,
                     "D_RFF": D_RFF, "features": FEATURE_COLS},
                    base + "_rff_params.joblib")
        joblib.dump(mu, base + "_rff_mu.joblib")
        print(f"[Saved] scaler/rff_params/rff_mu next to output")

if __name__ == "__main__":
    main()
