from Calc.basic_calc import (
    Add,
    Avg,
    Concatenation,
    Div,
    GeometricMean,
    MaxPool,
    MinPool,
    Mix,
    Mul,
    PostAdd,
    PostAngle,
    PostConcatenation,
    PostDiv,
    PostDivBy,
    PostGeometricMean,
    PostMaxPool,
    PostMinPool,
    PostMix,
    PostMul,
    PostSub,
    PostSubFrom,
    StdSub,
    Sub,
)
from Calc.complex_calc import (
    ComplexAdd,
    ComplexAngleMerge,
    ComplexMix,
)
from Calc.custom_calc import WidenMerge
from Calc.norm_calc import MatchStdMean, NormAngleMerge, NormStdMean, ProcStdMean
from Calc.process_calc import GitReBasin, QuantileMatch
from Calc.qeic_calc import QeicAdd, QeicMix, QeicSub

OPERATION_DICTS = {
    "add": Add,
    "sub": Sub,
    "mul": Mul,
    "div": Div,
    "mix": Mix,
    "avg": Avg,
    "concat": Concatenation,
    "maxpool": MaxPool,
    "minpool": MinPool,
    "geometric_mean": GeometricMean,
    "std_sub": StdSub,
    "widen": WidenMerge,
    "complexadd": ComplexAdd,
    "complex_angle_merge": ComplexAngleMerge,
    "complex_mix": ComplexMix,
    "qeic_add": QeicAdd,
    "qeic_mix": QeicMix,
    "qeic_sub": QeicSub,
}

# --- Preprocessing Dictionaries ---
PREPROCESS_DICT = {
    "none": lambda x, y, **kwargs: (x, y),  # Passthrough
    "pcd": GitReBasin,  # process_calc から
}


# --- Normalization Dictionaries ---
NORMALIZATION_DICT = {
    "none": lambda tensor,
    normalization_type,
    original_tensor,
    base_processed_values,
    *args,
    **kwargs: tensor,  # Passthrough
    "norm_std_mean": lambda tensor,
    normalization_type,
    original_tensor,
    base_processed_values,
    *args,
    **kwargs: NormStdMean(
        lambda v1, v2s, velocity: tensor, original_tensor, base_processed_values, 0
    ),
    "match_std_mean": lambda tensor,
    normalization_type,
    original_tensor,
    base_processed_values,
    *args,
    **kwargs: MatchStdMean(
        lambda v1, v2s, velocity: tensor, original_tensor, base_processed_values, 0
    ),
    "proc_std_mean": lambda tensor,
    normalization_type,
    original_tensor,
    base_processed_values,
    *args,
    **kwargs: ProcStdMean(
        lambda v1, v2s, velocity: tensor, original_tensor, base_processed_values, 0
    ),
    "angle_merge": lambda tensor,
    normalization_type,
    original_tensor,
    base_processed_values,
    *args,
    **kwargs: NormAngleMerge(
        lambda v1, v2s, velocity: tensor, original_tensor, base_processed_values, 0
    ),
    "quantile_match": lambda tensor,
    normalization_type,
    original_tensor,
    base_processed_values,
    *args,
    **kwargs: QuantileMatch(
        lambda v1, v2s, velocity: tensor,
        original_tensor,
        base_processed_values,
        0,  # velocityは使わない
        **kwargs,
    ),
}

# --- Post-processing Dictionaries ---
POST_OPERATION_DICT = {
    "add": PostAdd,
    "sub": PostSub,
    "subfrom": PostSubFrom,
    "mul": PostMul,
    "div": PostDiv,
    "divby": PostDivBy,
    "mix": PostMix,
    "concat": PostConcatenation,
    "maxpool": PostMaxPool,
    "minpool": PostMinPool,
    "geometric_mean": PostGeometricMean,
    "angle": PostAngle,
}
