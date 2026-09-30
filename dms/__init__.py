"""DMS library — controllers, policies, matching, metrics, proxies, inference."""
from __future__ import annotations

__version__ = "0.1.0"

from .controllers import (
    Hysteresis,
    RollingPercentile,
    SwitchStabiliser,
)
from .inference import (
    Detections,
    InferenceBackend,
    UltralyticsYOLOBackend,
    time_inference,
)
from .matching import (
    Match,
    MatchResult,
    box_iou,
    filter_by_score,
    greedy_match,
    iou_matrix,
)
from .metrics import (
    benefit_positive_frame,
    count_agree_frame,
    det_coverage_frame,
    det_recall_frame,
    iou_match_frame,
    latency_summary,
    summarise_run,
)
from .policies import (
    CANONICAL_POLICIES,
    POLICY_REGISTRY,
    Combined,
    CombinedHyst,
    ConfEMA,
    EntropyOnly,
    EXPERIMENTAL_POLICIES,
    FrameFeatures,
    LocalContrastHyst,
    MultiProxy,
    NOnly,
    Policy,
    PolicyConfig,
    SOnly,
    make_policy,
)
from .proxies import (
    ProxyConfig,
    bright_fraction,
    color_entropy,
    compute_proxies,
    edge_density,
    hue_std,
    laplacian_variance,
    local_contrast,
    shannon_entropy,
    tenengrad,
)

__all__ = [
    # controllers
    "Hysteresis",
    "RollingPercentile",
    "SwitchStabiliser",
    # inference
    "Detections",
    "InferenceBackend",
    "UltralyticsYOLOBackend",
    "time_inference",
    # matching
    "Match",
    "MatchResult",
    "box_iou",
    "filter_by_score",
    "greedy_match",
    "iou_matrix",
    # metrics
    "benefit_positive_frame",
    "count_agree_frame",
    "det_coverage_frame",
    "det_recall_frame",
    "iou_match_frame",
    "latency_summary",
    "summarise_run",
    # policies
    "FrameFeatures",
    "PolicyConfig",
    "Policy",
    "NOnly",
    "SOnly",
    "EntropyOnly",
    "Combined",
    "CombinedHyst",
    "LocalContrastHyst",
    "ConfEMA",
    "MultiProxy",
    "POLICY_REGISTRY",
    "CANONICAL_POLICIES",
    "EXPERIMENTAL_POLICIES",
    "make_policy",
    # proxies
    "ProxyConfig",
    "compute_proxies",
    "laplacian_variance",
    "shannon_entropy",
    "color_entropy",
    "tenengrad",
    "edge_density",
    "local_contrast",
    "bright_fraction",
    "hue_std",
]
