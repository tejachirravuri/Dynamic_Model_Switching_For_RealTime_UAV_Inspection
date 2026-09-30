"""Schema regression tests for future sweep outputs."""
from __future__ import annotations

from experiments.run_sweep import CSV_FIELDS, FRAME_FEATURES_FIELDS


REQUIRED_COLUMNS = {
    # Metadata
    "video_id", "video_filename", "domain", "device",
    "model_family", "pair_type", "frame_idx", "raw_video_idx",
    "timestamp_sec", "imgsz", "proxy_size", "preserve_aspect",
    "conf_floor", "iou_threshold",
    # Detector summaries
    "n_count", "s_count", "n_mean_conf", "n_max_conf",
    "s_mean_conf", "s_max_conf", "fast_mean_conf",
    "fast_ema", "slow_ema", "conf_drop",
    # Proxy values
    "laplacian", "entropy", "combined_score", "color_entropy",
    "tenengrad", "edge_density", "local_contrast", "bright_fraction",
    "hue_std", "brightness_mean", "brightness_std", "contrast",
    # Timing
    "proxy_ms_total", "proxy_ms_policy_relevant",
    "proxy_ms_laplacian", "proxy_ms_entropy",
    "proxy_ms_color_entropy", "proxy_ms_tenengrad",
    "proxy_ms_edge_density", "proxy_ms_local_contrast",
    "proxy_ms_brightness", "proxy_ms_hsv", "decision_ms",
    "n_latency_ms", "s_latency_ms", "chosen_latency_ms",
    "t_total_selected_ms", "t_total_deployed_ms",
    "t_total_shared_proxy_ms", "t_total_policy_relevant_ms",
    # Policy diagnostics
    "policy", "choice", "s_choice", "policy_score",
    "policy_thresh_low", "policy_thresh_high", "policy_thresh_mid",
    # Outcomes
    "iou_match", "det_coverage", "det_recall", "count_agree",
    "benefit_positive",
}


def test_sweep_per_frame_schema_contains_required_columns():
    assert REQUIRED_COLUMNS <= set(CSV_FIELDS)


def test_frame_features_schema_contains_required_columns():
    assert REQUIRED_COLUMNS <= set(FRAME_FEATURES_FIELDS)


def test_frame_features_has_no_policy_duplication_key_first():
    assert FRAME_FEATURES_FIELDS[:14] == (
        "video_id", "video_filename", "domain", "device",
        "model_family", "pair_type", "frame_idx", "raw_video_idx",
        "timestamp_sec", "imgsz", "proxy_size", "preserve_aspect",
        "conf_floor", "iou_threshold",
    )
