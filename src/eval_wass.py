from dataclasses import dataclass, field
from typing import Optional
from Bio.SeqUtils.ProtParam import ProteinAnalysis, ProtParamData
from dataclasses import asdict, dataclass, fields
import warnings
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import minmax_scale
@dataclass
class Descriptors:
    @classmethod
    def descriptor_names(cls) -> list[str]:
        return [f.name for f in fields(cls) if "not_a_descriptor" not in f.metadata]
    def asdict(self) -> dict:
        return asdict(self)
def _get_avg_quantity(sequence: str, lookup: dict[str, float]) -> float:
    return sum(lookup.get(aa, 0.0) for aa in sequence) / len(sequence)
@dataclass
class LargeMoleculeDescriptors(Descriptors):
    sequence: str = field(metadata={"not_a_descriptor": True})
    _protein_analysis: ProteinAnalysis = field(init=False, metadata={"not_a_descriptor": True})
    length: int = field(init=False)
    molecular_weight: float = field(init=False)
    aromaticity: float = field(init=False)
    instability_index: float = field(init=False)
    isoelectric_point: float = field(init=False)
    gravy: float = field(init=False)
    charge_at_pH6: float = field(init=False)
    charge_at_pH7: float = field(init=False)
    helix_fraction: float = field(init=False)
    turn_structure_fraction: float = field(init=False)
    sheet_structure_fraction: float = field(init=False)
    molar_extinction_coefficient_reduced: float = field(init=False)
    molar_extinction_coefficient_oxidized: float = field(init=False)
    avg_hydrophilicity: float = field(init=False)
    avg_surface_accessibility: float = field(init=False)
    def __post_init__(self):
        self.length = len(self.sequence)
        self._protein_analysis = ProteinAnalysis(self.sequence)
        self.molecular_weight = self._protein_analysis.molecular_weight()
        self.aromaticity = self._protein_analysis.aromaticity()
        self.instability_index = self._protein_analysis.instability_index()
        self.isoelectric_point = self._protein_analysis.isoelectric_point()
        self.gravy = self._protein_analysis.gravy()
        self.charge_at_pH6 = self._protein_analysis.charge_at_pH(6)
        self.charge_at_pH7 = self._protein_analysis.charge_at_pH(7)
        (
            self.helix_fraction,
            self.turn_structure_fraction,
            self.sheet_structure_fraction,
        ) = self._protein_analysis.secondary_structure_fraction()
        (
            self.molar_extinction_coefficient_reduced,
            self.molar_extinction_coefficient_oxidized,
        ) = self._protein_analysis.molar_extinction_coefficient()
        self.avg_hydrophilicity = _get_avg_quantity(self.sequence, ProtParamData.hw)
        self.avg_surface_accessibility = _get_avg_quantity(self.sequence, ProtParamData.em)
    @classmethod
    def from_sequence(cls, sequence: str) -> Optional["LargeMoleculeDescriptors"]:
        if len(sequence) > 0:
            return LargeMoleculeDescriptors(sequence)
        else:
            return None
class MetricWarning(RuntimeWarning):
    pass
LARGE_MOL_FIGSIZE = (3, 5)
NO_VALID_DESIGNS_WARNING = "There were no valid designs."
warnings.simplefilter("always", category=MetricWarning)
@dataclass
class MetricColumnInfo:
    feature_columns: list[str]
    sample_column: str
    figshape: tuple[int, int]
    def __post_init__(self):
        self.feature_columns = sorted(set(self.feature_columns))
def get_column_info(chain) -> MetricColumnInfo:
    if chain == "fv_heavy":
        return MetricColumnInfo(
            [f"fv_heavy_{feature}" for feature in LargeMoleculeDescriptors.descriptor_names()],
            "fv_heavy_aho",
            LARGE_MOL_FIGSIZE,
        )
    elif chain == "fv_light":
        return MetricColumnInfo(
            [f"fv_light_{feature}" for feature in LargeMoleculeDescriptors.descriptor_names()],
            "fv_light_aho",
            LARGE_MOL_FIGSIZE,
        )
    else:
        raise ValueError(f"Unknown chain type: {chain}")
def get_batch_descriptors(
    sample_df: pd.DataFrame, ref_feats: pd.DataFrame, chain
) -> tuple[dict[str, float], float, float, float]:
    """
    Compute aggregate statistics for a collection of samples compared to reference.
    Parameters
    ----------
    sample_df: pd.DataFrame
        Collection of samples, generally this would be a return value of
        `walkjump.callbacks.sample_and_compute_metrics()`
    ref_feats: pd.DataFrame
        Pre-computed reference distributions
    chain: ReferenceChainType
        Type of input molecule. Behavior switches based on molecule type.
    Returns
    -------
    Tuple[Dict[str, float], float, float, float]
        Wasserstein distances per statistic column and the
            (average wass. dist., total wass. dist., proportion not NaN)
    """
    info = get_column_info(chain)
    try:
        prop_valid = float(sample_df[info.sample_column].notna().sum()) / len(sample_df)
    except ZeroDivisionError:
        warnings.warn(NO_VALID_DESIGNS_WARNING, category=MetricWarning, stacklevel=2)
        prop_valid = 0.0
    wasserstein_distances = {}
    for column in info.feature_columns:
        # filter out NaN rows for this column
        valid = sample_df.loc[sample_df[column].notna(), column]
        valid_ref = ref_feats.loc[ref_feats[column].notna(), column]
        # min/max norm the validated rows.
        try:
            normed = minmax_scale(valid)
            normed_ref = minmax_scale(valid_ref)
            # compute wasserstein
            wasserstein_distances[f"{column}_wd"] = wasserstein_distance(normed, normed_ref)
        except ValueError:
            wasserstein_distances[f"{column}_wd"] = float("inf")
    
    total_wd = sum(wasserstein_distances.values())
    std_wd = np.std(np.array(list(wasserstein_distances.values())))
    avg_wd = total_wd / len(info.feature_columns)
    return wasserstein_distances, std_wd,avg_wd, total_wd, prop_valid
mode_to_data_dir = {
    "train": "/home/srinu_pd/walk-jump-poas/data/poas.csv.gz",
}
def main():
    df = (
        pd.read_csv("/home/srinu_pd/walk-jump-poas/samples_0.5_10k.csv")
    )
    print("loaded generated data")
    dataset = pd.read_csv(mode_to_data_dir["train"], compression="gzip")
    train_df = dataset[dataset.partition == "val"]
    #test_df = dataset[dataset.partition == "test"]
    ref_seqs = [heavy+light for heavy, light in zip(train_df.fv_heavy_aho, train_df.fv_light_aho)]
    ref_seqs = [seq.replace("-", "") for seq in ref_seqs]
    #print(ref_seqs[0])
    #refs = ref_seqs[:1000]
    
    sample_seqs = [heavy+light for heavy, light in zip(df.fv_heavy_aho, df.fv_light_aho)]
    sample_seqs = [seq.replace("-", "") for seq in sample_seqs]
    desc_names = LargeMoleculeDescriptors.descriptor_names()
    def compute_descriptors(sequences, prefix=""):
        descriptors_list = []
        for seq in sequences:
            desc = LargeMoleculeDescriptors.from_sequence(seq)
            #print(desc.length)
            if desc is None:
                # fill None for all descriptors
                descriptors_list.append({f"{prefix}{feature}": None for feature in desc_names})
                continue
            # add prefix to each feature
            descriptors_list.append({f"{prefix}{feature}": getattr(desc, feature) for feature in desc_names})
        return pd.DataFrame(descriptors_list)
    # Create descriptor DataFrames
    ref_feats = compute_descriptors(ref_seqs, prefix="fv_heavy_")
    sample_feats = compute_descriptors(sample_seqs, prefix="fv_heavy_")
    # Add the sequence column
    ref_feats["fv_heavy_aho"] = ref_seqs
    sample_feats["fv_heavy_aho"] = sample_seqs
    wd_dict, std_wd,avg_wd, total_wd, prop_valid = get_batch_descriptors(
        ref_feats, sample_feats, chain="fv_heavy"
    )
    print("Per-column Wasserstein distances:\n", wd_dict)
    print("Std WD:", std_wd)
    print("Average WD:", avg_wd)
    print("Total WD:", total_wd)
    print("Proportion valid samples:", prop_valid)
if __name__ == "__main__":
    main()