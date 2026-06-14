# script to perform distributional conformity score (DCS) evaluation
from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np
from sklearn.neighbors import KernelDensity
from Bio.SeqUtils.ProtParam import ProteinAnalysis, ProtParamData
import editdistance
from sklearn.preprocessing import StandardScaler
from eval_wass import LargeMoleculeDescriptors
from sklearn.preprocessing import minmax_scale
#from awkde import AKDE
# use argparse to get the choise of ggdwjs or dwjs
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="/home/srinu_pd/walk-jump-poas/samples_ours_10k.csv", help="ggdwjs_beta, ggdwjs_ii, ggdwjs_beta_ii, dwjs, train")

directory = parser.parse_args().dir
print("From: ", directory)


mode_to_data_dir = {
    "train": "/data0/srinu_pd/walk-jump/data/poas.csv.gz",
}

# get training data
dataset = pd.read_csv(mode_to_data_dir["train"], compression="gzip")
train_df = dataset[dataset.partition == "val"] #train
test_df = dataset[dataset.partition == "train"] #val

ref_seqs = [heavy+light for heavy, light in zip(train_df.fv_heavy_aho, train_df.fv_light_aho)]
print(ref_seqs[0:2])
ref_seqs = [seq.replace("-", "") for seq in ref_seqs]

test_seqs = [heavy+light for heavy, light in zip(test_df.fv_heavy_aho, test_df.fv_light_aho)]
test_seqs = [seq.replace("-", "") for seq in test_seqs[0:1000]]
N = len(test_seqs)
print("N=",N)
# get the data
df = (
    pd.read_csv(directory)
    if not "gz" in directory
    else pd.read_csv(directory, compression="gzip")
)
sample_seqs = [heavy+light for heavy, light in zip(df.fv_heavy_aho, df.fv_light_aho)]
sample_seqs = [seq.replace("-", "") for seq in ref_seqs[0:500]]

# # get protein analysis object for each sequence
# from Bio.SeqUtils.ProtParam import ProteinAnalysis

# sequences = [ProteinAnalysis(str(seq)) for seq in sample_seqs]

# # get a list of the beta sheet percentages
# beta_sheets = np.array([seq.secondary_structure_fraction()[2] for seq in sequences])
# instability_indices = np.array([seq.instability_index() for seq in sequences])
# aromaticity = np.array([seq.aromaticity() for seq in sequences])

# print(f"beta sheet percentages: {beta_sheets.mean()} +- {beta_sheets.std()}")
# print(f"instability indices: {instability_indices.mean()} +- {instability_indices.std()}")
# print(f"aromaticity: {aromaticity.mean()} +- {aromaticity.std()}")



# Function to calculate edit distance between two sequences
def calculate_edit_distance(seq1, seq2):
    return editdistance.eval(seq1, seq2)

# Function to calculate average edit distance
def unique_percentage(sequences):
    unique = set(sequences)
    return len(unique) / len(sequences)

def intra_div(sequences):
    total_distance = 0
    num_pairs = 0
    
    edit_distances = []
    # Calculate total distance
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            edit_distances.append(calculate_edit_distance(sequences[i], sequences[j]))
    
    edit_distances = np.array(edit_distances)
    return edit_distances.mean(), edit_distances.std()



print(f"Unique percentage: {unique_percentage(sample_seqs)}")
ed = intra_div(sample_seqs)
print(f"Intra Diversity: {ed[0]} +- {ed[1]}")



k = 1000#len(test_seqs) # number of samples to draw from the test set
n = 10000#len(ref_seqs) # number of samples to draw from the reference/training set


tests =  test_seqs[: k - 1]
refs = ref_seqs[:n]

#tests = test_seqs[int(N*0.8):] 
#refs = test_seqs[:int(N*0.8)]

print("number of validation samples",len(tests))
print("number of trainign samples",len(refs))
def edit_distance(sequences,normalize=False):
    
    min_dists = np.empty(len(sequences), dtype=np.float64)

    for i, g in enumerate(sequences):
        # Compute distance to all reference sequences
        dists = [calculate_edit_distance(g, r) for r in tests]
        
        if normalize:
            # Divide by max sequence length to normalize distance to [0,1]
            dists = [d / max(len(g), len(r)) if max(len(g), len(r)) > 0 else 0 for d, r in zip(dists, tests)]

        # Take the minimum distance (nearest neighbor)
        min_dists[i] = np.min(dists)

    return float(min_dists.mean()), float(min_dists.std())

ed = edit_distance(sample_seqs)
print(f"Edit Distance: {ed[0]} +- {ed[1]}")

def _get_avg_quantity(sequence: str, lookup: dict[str, float]) -> float:
    return sum(lookup.get(aa, 0.0) for aa in sequence) / len(sequence)

def featurize_sequences(seqs):
    """
    seqs: list/array of strings (amino-acid sequences)
    returns: (N,2) float array: [GRAVY, molecular_weight]
    """
    feats = []
    for s in seqs:
        # ensure string and uppercase; strip spaces/newlines
      
        pa = ProteinAnalysis(s)
        helix, turn, sheet = pa.secondary_structure_fraction()
        molar_extin_red, molar_extin_ox = pa.molar_extinction_coefficient()
        feats.append([float(pa.molecular_weight()),float(pa.gravy()),
            # float(pa.isoelectric_point()),    # pI
            #     float(pa.aromaticity()),          # aromaticity
            #     float(pa.instability_index()), 
            #     sheet,
            #     float(pa.charge_at_pH(6)),
            #     float(pa.charge_at_pH(67)),
            #     float(helix),
            #     float(turn),
            #     float(molar_extin_red),
            #     float(molar_extin_ox),
            #     float(_get_avg_quantity(s, ProtParamData.hw)),
            #     float(_get_avg_quantity(s, ProtParamData.em))
                ])
    return np.asarray(feats, dtype=np.float64)

def _silverman_bandwidth(X: np.ndarray) -> float:
    n, d = X.shape
    std = np.std(X, axis=0, ddof=1)
    return ((4 / (d + 2)) ** (1 / (d + 4))) * (n ** (-1 / (d + 4))) * np.mean(std)

    
    
    
def get_kde(refs, bandwidth=0.15):
    """
    Fit a Gaussian KDE on z-scored (GRAVY, MW) from reference sequences.
    Returns (kde, scaler).
    """
    # features
    #ref_hydros = [ProteinAnalysis(str(r)).gravy() for r in refs]
    #ref_mw     = [ProteinAnalysis(str(r)).molecular_weight() for r in refs]
    #X = np.column_stack([ref_hydros, ref_mw]).astype(np.float64)
    X = featurize_sequences(refs)
    # z-score so both dims contribute equally
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    

    bw = _silverman_bandwidth(Xs)
    kde = KernelDensity(kernel="gaussian", bandwidth=bw * (1 + bandwidth)).fit(Xs)

    # ---- fixed bandwidth (on standardized space) ----
    #kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(Xs)
    #kde = AKDE(bw="silverman", kernel="gaussian", diag=True, alpha=bandwidth)

    # # ---- (optional) small grid search on standardized space ----
    # grid = np.geomspace(0.05, 1.5, 10)
    # search = GridSearchCV(KernelDensity(kernel="gaussian"),
    #                       {"bandwidth": list(grid)}, cv=5, n_jobs=-1)
    # search.fit(Xs)
    # kde = search.best_estimator_
    # print("Best bandwidth:", search.best_params_['bandwidth'])

    return kde, scaler


# def conformity_score(sample:str, ref_kde,scaler):
#     """Computes the conformity score of a sample with respect to a set of reference sequences.

#     Parameters
#     ----------
#     sample : str
#         The sample sequence.
#     refs : list[str]
#         The reference sequences.

#     Returns
#     -------
#     float
#         The conformity score.
#     """
#     # get the hydrophilicity and molecular weight of the sample
#     #sample_hydro = ProteinAnalysis(str(sample)).gravy()
#     #sample_mol_weight = ProteinAnalysis(str(sample)).molecular_weight()

#     pa = ProteinAnalysis(sample)
#     helix, turn, sheet = pa.secondary_structure_fraction()
#     x = np.array([[float(pa.gravy()), 
#                     float(pa.molecular_weight()),
#                     float(pa.isoelectric_point()),    # pI
#                 float(pa.aromaticity()),          # aromaticity
#                 float(pa.instability_index()), 
#                 sheet
#                 ]], dtype=np.float64)  # shape (1,2)
#     #print(x)
#     if scaler is not None:
#       x = scaler.transform(x)

#     logp = ref_kde.score_samples(x)[0]
#     return logp
    # # get the log probability of the sample
    # log_prob = ref_kde.score_samples(np.array([sample_hydro, sample_mol_weight]).reshape(-1, 2))
    # # get the conformity score
    # # print(log_prob)
    # conformity_score = log_prob[
    #     0
    # ]  # the log probability of the sample is the first element of the log_prob array (high log probability means high conformity)
    # return conformity_score
from scipy.linalg import sqrtm

def fid(scaler):
    #ref_hydros = [ProteinAnalysis(str(ref)).gravy() for ref in refs]
    #ref_mol_weights = [ProteinAnalysis(str(ref)).molecular_weight() for ref in refs]
    # ref_features = np.array([
    #     [ProteinAnalysis(str(test)).gravy(), ProteinAnalysis(str(test)).molecular_weight()]
    #     for test in tests
    # ])
    ref_features = featurize_sequences(tests)
    #print(ref)
    if scaler is not None:
       features1 = scaler.transform(ref_features)
    else:
        features1 = ref_features
    # samp_features = np.array([
    #     [ProteinAnalysis(str(samp)).gravy(), ProteinAnalysis(str(samp)).molecular_weight()]
    #     for samp in sample_seqs
    # ])
    samp_features = featurize_sequences(sample_seqs)
    if scaler is not None:
       features2 = scaler.transform(samp_features)
    else:
        features2 = samp_features

    # Mean and covariance
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
    # Mean difference
    diff = mu1 - mu2
    diff_sq = diff @ diff
    # Covariance sqrt
    covmean = sqrtm(sigma1 @ sigma2)
    # Handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # FID
    fid = diff_sq + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


ref_kde,scaler = get_kde(refs)
#print("FID",fid(scaler))
scores = []
# for sample_id, sample in tqdm(enumerate(sample_seqs)):
#     # add sample to the testidation set (get a new list)
#     target_tests = tests + [sample]
#     conformity_scores = []
#     for i, test in enumerate(target_tests):
#         conformity_scores.append(conformity_score(test, ref_kde,scaler))

#     # find how many of the conformity scores are greater than the conformity score of the sample
#     conformity_scores = np.array(conformity_scores)
#     num_less_than_sample = (conformity_scores < conformity_scores[-1]).sum()
#     # print(f"DCS: {num_less_than_sample / k}")
#     scores.append(num_less_than_sample / k)

# print(f"DCS: {np.mean(scores)} +- {np.std(scores)}")

# ---------- 1) Featurize sequences (vectorized over list) ----------


# ---------- 2) Batch conformity scores given 2-D features ----------
def conformity_score_batch_features(X_feat, ref_kde, scaler=None):
    """
    X_feat: (N,2) numeric features (already featurized)
    ref_kde: sklearn KernelDensity-like with .score_samples
    returns: (N,) nonconformity scores (here = - log p)
    """
    X2 = scaler.transform(X_feat) if scaler is not None else X_feat
    #if hasattr(ref_kde, "score_samples"):
    logp = ref_kde.score_samples(X2)   # shape (N,)
    return logp                       # nonconformity = -log p
    # Fallback (rare): row-wise
    #return np.array([ref_kde.score_samples(x[None, :])[0] for x in X2])

# ---------- 3) Fast DCS ----------
def fast_dcs(tests, sample_seqs, ref_kde, scaler=None, tie_rule="left"):
    """
    tests: testidation sequences (list[str])
    sample_seqs: generated sequences (list[str])
    tie_rule: 'left' (strict <), or 'ties_half' for randomized conformal
    returns: scores (N,), mean, std
    """
    # Featurize once
    tests_feat    = featurize_sequences(tests)          # (M,2)
    samples_feat = featurize_sequences(sample_seqs)   # (N,2)

    # Precompute test scores once
    test_scores = conformity_score_batch_features(tests_feat, ref_kde, scaler)  # (M,)
    test_sorted = np.sort(test_scores)
    k = len(test_scores) + 1

    # All sample scores at once
    sample_scores = conformity_score_batch_features(samples_feat, ref_kde, scaler)  # (N,)

    # Count how many test scores are < sample score
    if tie_rule == "left":
        counts = np.searchsorted(test_sorted, sample_scores, side="left")
        dcs = counts / k
    # else:  # ties_half (optional randomized conformal)
    #     left  = np.searchsorted(test_sorted, sample_scores, side="left")
    #     right = np.searchsorted(test_sorted, sample_scores, side="right")
    #     ties  = right - left
    #     dcs   = (left + 0.5 * ties) / k

    return dcs, dcs.mean(), dcs.std()

#---------- Example use ----------
dcs, mu, sd = fast_dcs(tests, sample_seqs, ref_kde, scaler, tie_rule="left")
print(f"DCS: {mu:.6f} +- {sd:.6f}")



