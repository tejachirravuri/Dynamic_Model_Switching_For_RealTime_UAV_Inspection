"""Unit tests for dms.proxies — scene-feature computation.

Audit checklist:
  - proxy_size is a first-class flag (different sizes -> different values)
  - preserve_aspect letterboxes vs warps
  - constant-input edge cases (all features = 0)
  - timing output is positive
  - unknown feature names raise
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from dms.proxies import (
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


# ===========================================================================
# Synthetic image helpers
# ===========================================================================
def _gray_constant(value=128, h=128, w=128) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _bgr_constant(b=64, g=128, r=192, h=128, w=128) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def _gray_random(h=128, w=128, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr_random(h=128, w=128, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _gray_horizontal_gradient(h=128, w=128) -> np.ndarray:
    """Per-column intensity = column index, scaled into [0, 255]."""
    col = (np.arange(w) * (255 // max(w - 1, 1))).astype(np.uint8)
    return np.tile(col, (h, 1))


# ===========================================================================
# Per-feature primitives
# ===========================================================================
class TestLaplacianVariance:
    def test_constant_zero(self):
        assert laplacian_variance(_gray_constant()) == pytest.approx(0.0)

    def test_random_positive(self):
        assert laplacian_variance(_gray_random()) > 0


class TestShannonEntropy:
    def test_constant_zero(self):
        assert shannon_entropy(_gray_constant()) == pytest.approx(0.0)

    def test_random_near_uniform_high(self):
        # 256x256 uniform-random ~ uniform 8-bit -> entropy near 8 bits.
        img = _gray_random(h=256, w=256)
        H = shannon_entropy(img)
        assert 7.0 < H < 8.0

    def test_two_value_image(self):
        # Half 0, half 255 -> entropy = 1 bit (binary uniform).
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255
        H = shannon_entropy(img)
        assert H == pytest.approx(1.0, abs=1e-6)


class TestColorEntropy:
    def test_constant_zero(self):
        assert color_entropy(_bgr_constant()) == pytest.approx(0.0)

    def test_random_positive(self):
        assert color_entropy(_bgr_random()) > 0

    def test_grayscale_input_returns_zero(self):
        # A 2D array passed to color_entropy returns 0 (defensive guard).
        assert color_entropy(_gray_constant()) == pytest.approx(0.0)


class TestTenengrad:
    def test_constant_zero(self):
        assert tenengrad(_gray_constant()) == pytest.approx(0.0)

    def test_gradient_positive(self):
        assert tenengrad(_gray_horizontal_gradient()) > 0


class TestEdgeDensity:
    def test_constant_zero(self):
        assert edge_density(_gray_constant()) == pytest.approx(0.0)

    def test_value_in_unit_interval(self):
        # Any image's edge density is a fraction in [0, 1].
        img = _gray_random()
        d = edge_density(img)
        assert 0.0 <= d <= 1.0


class TestLocalContrast:
    def test_constant_zero(self):
        assert local_contrast(_gray_constant()) == pytest.approx(0.0)

    def test_random_positive(self):
        assert local_contrast(_gray_random()) > 0


class TestBrightFraction:
    def test_constant_dark_zero(self):
        img = _gray_constant(value=0)
        assert bright_fraction(img) == pytest.approx(0.0)

    def test_constant_bright_one(self):
        img = _gray_constant(value=250)
        assert bright_fraction(img, threshold=200) == pytest.approx(1.0)

    def test_threshold_boundary_inclusive(self):
        img = _gray_constant(value=200)
        assert bright_fraction(img, threshold=200) == pytest.approx(1.0)


class TestHueStd:
    def test_constant_zero(self):
        assert hue_std(_bgr_constant()) == pytest.approx(0.0)

    def test_random_positive(self):
        assert hue_std(_bgr_random()) > 0

    def test_grayscale_returns_zero(self):
        assert hue_std(_gray_constant()) == pytest.approx(0.0)


# ===========================================================================
# compute_proxies driver
# ===========================================================================
class TestComputeProxies:
    def test_returns_all_requested_features_and_metadata(self):
        img = _bgr_random(h=200, w=200)
        cfg = ProxyConfig(
            size=64,
            preserve_aspect=False,
            features=("L", "H", "tenengrad", "color_entropy"),
        )
        out = compute_proxies(img, cfg)
        for k in ("L", "H", "tenengrad", "color_entropy", "proxy_size", "time_ms"):
            assert k in out
        assert out["proxy_size"] == 64

    def test_timing_is_positive(self):
        img = _bgr_random()
        out = compute_proxies(img, ProxyConfig(size=160))
        assert out["time_ms"] > 0

    def test_proxy_size_changes_features(self):
        # Same input + different proxy_size -> different feature values.
        # This is the methodological point: small proxies discard
        # information.
        img = _bgr_random(h=512, w=512, seed=42)
        cfg_small = ProxyConfig(size=32, features=("L",))
        cfg_large = ProxyConfig(size=256, features=("L",))
        out_s = compute_proxies(img, cfg_small)
        out_l = compute_proxies(img, cfg_large)
        assert out_s["L"] != out_l["L"]

    def test_native_size_skips_resize(self):
        img = _bgr_random(h=100, w=100, seed=1)
        cfg = ProxyConfig(size=0, features=("local_contrast",))
        out = compute_proxies(img, cfg)
        # local_contrast on the unresized image equals the direct call.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert out["local_contrast"] == pytest.approx(local_contrast(gray))

    def test_preserve_aspect_runs_on_non_square(self):
        # 1920x1080-style aspect; preserve_aspect=True must letterbox
        # without raising and without distortion.
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[:540, :960] = 255  # solid block in upper-left quadrant
        cfg = ProxyConfig(size=160, preserve_aspect=True,
                          features=("L", "H"))
        out = compute_proxies(img, cfg)
        assert "L" in out and "H" in out
        assert isinstance(out["L"], float)

    def test_preserve_aspect_differs_from_warp(self):
        # Same non-square input under both modes -> features differ.
        img = _bgr_random(h=300, w=600, seed=7)
        cfg_warp = ProxyConfig(size=128, preserve_aspect=False,
                               features=("L",))
        cfg_letter = ProxyConfig(size=128, preserve_aspect=True,
                                 features=("L",))
        out_w = compute_proxies(img, cfg_warp)
        out_l = compute_proxies(img, cfg_letter)
        assert out_w["L"] != out_l["L"]

    def test_unknown_feature_raises(self):
        img = _bgr_constant()
        cfg = ProxyConfig(size=64, features=("does_not_exist",))
        with pytest.raises(ValueError, match="unknown feature"):
            compute_proxies(img, cfg)

    def test_constant_image_all_zeros_for_variance_features(self):
        img = _bgr_constant(b=128, g=128, r=128)
        cfg = ProxyConfig(
            size=64, preserve_aspect=False,
            features=("L", "H", "tenengrad", "edge_density",
                      "local_contrast", "color_entropy"),
        )
        out = compute_proxies(img, cfg)
        assert out["L"] == pytest.approx(0.0)
        assert out["H"] == pytest.approx(0.0)
        assert out["tenengrad"] == pytest.approx(0.0)
        assert out["edge_density"] == pytest.approx(0.0)
        assert out["local_contrast"] == pytest.approx(0.0)
        assert out["color_entropy"] == pytest.approx(0.0)

    def test_default_features_match_thesis_sweep(self):
        cfg = ProxyConfig()
        # Hardened sweep schema defaults: all image-proxy values logged.
        assert set(cfg.features) == {
            "L", "H", "color_entropy", "tenengrad",
            "edge_density", "local_contrast",
            "bright_fraction", "hue_std", "brightness_mean",
        }
