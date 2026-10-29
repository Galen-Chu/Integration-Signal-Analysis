"""特徵工程測試——合成訊號、精確數值基準."""
import numpy as np

from src.isa.features.freq_domain import (band_energy_ratio, combine,
                                          dominant_frequency_hz,
                                          extract_freq_features,
                                          spectral_centroid_hz,
                                          spectral_entropy)
from src.isa.features.time_domain import (extract_time_features,
                                          sma_3d)

FS = 50
T = np.arange(128) / FS


def _sine(freq, amp=1.0):
    return amp * np.sin(2 * np.pi * freq * T)


class TestFreqDomain:
    def test_dominant_frequency_of_pure_sine(self):
        # 3Hz 正弦 → 主頻落最近 bin（bin 寬 50/128 ≈ 0.39Hz，容差 0.5）
        assert abs(dominant_frequency_hz(_sine(3.0), FS) - 3.0) < 0.5
        assert abs(dominant_frequency_hz(_sine(1.0), FS) - 1.0) < 0.5

    def test_dominant_frequency_ignores_dc(self):
        x = _sine(2.0) + 100.0   # 大 DC 偏移不可干擾主頻
        assert abs(dominant_frequency_hz(x, FS) - 2.0) < 0.5

    def test_band_energy_ratio(self):
        # 純 2Hz 訊號的能量應幾乎全落在 1-3Hz 頻帶
        assert band_energy_ratio(_sine(2.0), 1.0, 3.0, FS) > 0.95
        # 純 5Hz 訊號在 1-3Hz 頻帶應接近 0
        assert band_energy_ratio(_sine(5.0), 1.0, 3.0, FS) < 0.05

    def test_spectral_centroid_of_pure_sine(self):
        assert abs(spectral_centroid_hz(_sine(4.0), FS) - 4.0) < 0.5

    def test_spectral_entropy_bounds_and_ordering(self):
        # 規律正弦（低熵）< 混亂雜訊（高熵）
        sine_e = spectral_entropy(_sine(2.0), FS)
        noise_e = spectral_entropy(np.random.default_rng(0).normal(size=128), FS)
        assert 0 <= sine_e <= 1 and 0 <= noise_e <= 1
        assert sine_e < noise_e

    def test_extract_freq_features_shape(self):
        rng = np.random.default_rng(1)
        windows = rng.normal(size=(5, 9, 128)).astype(np.float32)
        feats = extract_freq_features(windows)
        assert feats.shape == (5, 16)
        assert np.isfinite(feats).all()


class TestTimeDomain:
    def test_sma_3d_exact(self):
        x = np.array([1.0, -2.0, 3.0])
        y = np.array([0.0, 0.0, 0.0])
        z = np.array([1.0, 1.0, 1.0])
        # (1+2+3) + 0 + 3 = 9 / 3 = 3.0
        assert sma_3d(x, y, z) == 3.0

    def test_extract_time_features_shape_and_names(self):
        from src.isa.features.time_domain import FEATURE_NAMES
        rng = np.random.default_rng(2)
        windows = rng.normal(size=(4, 9, 128)).astype(np.float32)
        feats = extract_time_features(windows)
        assert feats.shape == (4, len(FEATURE_NAMES))
        assert len(FEATURE_NAMES) == 4 * 6 + 3  # 4 訊號×6 統計 + 3 軸相關
        assert np.isfinite(feats).all()

    def test_mean_feature_matches_manual(self):
        windows = np.zeros((1, 9, 128), dtype=np.float32)
        windows[0, 0, :] = 2.0   # body_acc_x 全 2.0
        feats = extract_time_features(windows)
        # FEATURE_NAMES[0] = acc_x_mean
        assert feats[0, 0] == 2.0


def test_combine():
    a = np.zeros((3, 2), dtype=np.float32)
    b = np.ones((3, 4), dtype=np.float32)
    assert combine(a, b).shape == (3, 6)
