"""Scene-proxy computation — per-frame image statistics used by policies.

The thesis sweep computes proxies on a downsampled copy of each frame.
``proxy_size`` is a first-class parameter so the resolution ablation
(160 vs 320 vs 480 vs native) is a config flag, not a code change.

``preserve_aspect`` controls whether the resize letterboxes (preserves
aspect with padding) or warps to ``size x size``. The legacy thesis
sweep used warp-resize; ``preserve_aspect=True`` is provided as the
methodologically correct alternative for the ablation table.

Each call returns a dict with the requested feature values plus
``proxy_size`` and ``time_ms`` for downstream master-table aggregation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Sequence

import cv2
import numpy as np


# ===========================================================================
# Resize helpers
# ===========================================================================
def _letterbox(img: np.ndarray, size: int, pad_value: int = 114) -> np.ndarray:
    """Aspect-preserving resize to a ``size x size`` canvas with padding."""
    h, w = img.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas_shape = (size, size) + img.shape[2:]
    canvas = np.full(canvas_shape, pad_value, dtype=img.dtype)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def _resize_for_proxy(
    img: np.ndarray,
    size: int,
    preserve_aspect: bool,
) -> np.ndarray:
    """Resize ``img`` to ``size`` per the active aspect policy.

    ``size <= 0`` skips resize and returns the input unchanged (native-
    resolution proxy mode).
    """
    if size <= 0:
        return img
    if preserve_aspect:
        return _letterbox(img, size)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# ===========================================================================
# Per-feature implementations
# ===========================================================================
def laplacian_variance(img_gray: np.ndarray) -> float:
    """Variance of the Laplacian. Higher = sharper / more texture."""
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def shannon_entropy(img_gray: np.ndarray) -> float:
    """Per-pixel grayscale Shannon entropy in bits (256-bin histogram)."""
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / total
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def color_entropy(img_bgr: np.ndarray, bins: int = 16) -> float:
    """Joint Shannon entropy of HSV (H, S) channels in bits."""
    if img_bgr.ndim != 3:
        return 0.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].ravel()
    s = hsv[:, :, 1].ravel()
    hist, _, _ = np.histogram2d(
        h, s, bins=bins, range=[[0, 180], [0, 256]],
    )
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def tenengrad(img_gray: np.ndarray) -> float:
    """Mean of squared Sobel gradient magnitude. Higher = more edges."""
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    return float((gx * gx + gy * gy).mean())


def edge_density(
    img_gray: np.ndarray,
    low: int = 50,
    high: int = 150,
) -> float:
    """Fraction of pixels classified as edges by Canny."""
    edges = cv2.Canny(img_gray, low, high)
    return float((edges > 0).mean())


def local_contrast(img_gray: np.ndarray) -> float:
    """Standard deviation of pixel intensities (RMS contrast)."""
    return float(img_gray.astype(np.float64).std())


def bright_fraction(img_gray: np.ndarray, threshold: int = 200) -> float:
    """Fraction of pixels at or above ``threshold`` in 0..255."""
    return float((img_gray >= threshold).mean())


def brightness_mean(img_gray: np.ndarray) -> float:
    """Mean grayscale intensity (0..255). Cheap brightness proxy.

    Note: ``brightness_std`` is equivalent to :func:`local_contrast` (both
    are ``gray.std()``). The schema exposes both names; the runner wires
    them to the same callable.
    """
    return float(img_gray.astype(np.float64).mean())


def hue_std(img_bgr: np.ndarray) -> float:
    """Standard deviation of HSV hue channel.

    NOTE: hue is circular (0..180 in OpenCV). Linear std is an
    approximation; the thesis treats it as a coarse colour-spread proxy.
    """
    if img_bgr.ndim != 3:
        return 0.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 0].astype(np.float64).std())


# Map feature name -> callable. The callable accepts whichever colour
# variant is appropriate (gray vs BGR); the dispatcher in
# :func:`compute_proxies` selects the right input.
_GRAY_FEATURES = {
    "L": laplacian_variance,
    "H": shannon_entropy,
    "tenengrad": tenengrad,
    "edge_density": edge_density,
    "local_contrast": local_contrast,
    "bright_fraction": bright_fraction,
    "brightness_mean": brightness_mean,
}
_COLOR_FEATURES = {
    "color_entropy": color_entropy,
    "hue_std": hue_std,
}


# ===========================================================================
# Config + driver
# ===========================================================================
@dataclass
class ProxyConfig:
    """Configuration for :func:`compute_proxies`."""

    size: int = 160
    preserve_aspect: bool = False
    features: Sequence[str] = field(default_factory=lambda: (
        "L", "H", "color_entropy", "tenengrad",
        "edge_density", "local_contrast",
        "bright_fraction", "hue_std", "brightness_mean",
    ))


def compute_proxies(
    frame_bgr: np.ndarray,
    config: ProxyConfig,
) -> Dict[str, float]:
    """Compute the requested scene proxies for a single BGR frame.

    Args:
        frame_bgr: ``(H, W, 3)`` uint8 BGR image (typically the original
            video frame, full resolution).
        config:    :class:`ProxyConfig` selecting size, aspect mode, and
            which features to compute.

    Returns:
        dict containing one float per requested feature, plus:
          * ``"proxy_size"`` (the size actually used)
          * ``"time_ms"`` total wall-clock (resize+colour+all features)
          * ``"time_overhead_ms"`` resize + colour conversion only
          * ``"time_<feature>_ms"`` per-feature wall-clock (one entry per
            requested feature). Lets the master table report which
            feature dominates the proxy budget instead of just the sum.

    Raises:
        ValueError: if ``config.features`` contains an unknown name.
    """
    t0 = time.perf_counter()
    img = _resize_for_proxy(frame_bgr, config.size, config.preserve_aspect)
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    overhead_ms = float((time.perf_counter() - t0) * 1000.0)

    out: Dict[str, float] = {}
    for fname in config.features:
        t_f = time.perf_counter()
        if fname in _GRAY_FEATURES:
            out[fname] = _GRAY_FEATURES[fname](gray)
        elif fname in _COLOR_FEATURES:
            out[fname] = _COLOR_FEATURES[fname](img)
        else:
            raise ValueError(
                f"unknown feature: {fname!r}. "
                f"valid: {sorted(set(_GRAY_FEATURES) | set(_COLOR_FEATURES))}"
            )
        out[f"time_{fname}_ms"] = float((time.perf_counter() - t_f) * 1000.0)

    out["proxy_size"] = int(config.size)
    out["time_overhead_ms"] = overhead_ms
    out["time_ms"] = float((time.perf_counter() - t0) * 1000.0)
    return out


__all__ = [
    "ProxyConfig",
    "compute_proxies",
    "laplacian_variance",
    "shannon_entropy",
    "color_entropy",
    "tenengrad",
    "edge_density",
    "local_contrast",
    "bright_fraction",
    "brightness_mean",
    "hue_std",
]
