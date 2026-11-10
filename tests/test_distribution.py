"""Distribution features tests · ECDF 百分位與頻帶能量占比."""
import numpy as np

from src.isa.features.distribution import (
    BANDS_HZ,
    ECDF_QUANTILES,
    FEATURE_NAMES_BANDS,
    FEATURE_NAMES_ECDF,
    extract_band_features,
    extract_ecdf_features,
)


def _windows(body, total):
    """組 (1, 9, 128)：0-2 body、6-8 total，其餘填零."""
    w = np.zeros((1, 9, 128))
    w[0, 0:3, :] = body
    w[0, 6:9, :] = total
    return w


class TestEcdf:
    def test_names_count(self):
        assert len(FEATURE_NAMES_ECDF) == 6 * len(ECDF_QUANTILES) == 30

    def test_percentiles_of_known_ramp(self):
        # 0..127 線性斜坡；佈局＝訊號主序：[body_x p5..p95][body_y p5..p95]…
        x = np.arange(128, dtype=float)
        w = _windows(np.stack([x, x * 0, x * 0]), np.stack([x, x * 0, x * 0]))
        out = extract_ecdf_features(w)[0]
        qs = np.percentile(x, ECDF_QUANTILES)
        np.testing.assert_allclose(out[:5], qs, rtol=1e-6)   # body_x 五百分位
        assert out[5] == np.float32(0.0)                     # body_y（全零訊號）
        np.testing.assert_allclose(out[15:20], qs, rtol=1e-6)  # total_x
        idx_total_x_p50 = 3 * len(ECDF_QUANTILES) + 2
        assert out[idx_total_x_p50] == np.float32(qs[2])

    def test_multi_window_layout(self):
        # (N, 6, Q)：列＝視窗——抓轉置/reshape 錯序的守門測試
        w = np.zeros((2, 9, 128))
        w[0, 0, :] = 10.0
        w[1, 0, :] = 20.0
        out = extract_ecdf_features(w)
        assert out.shape == (2, 30)
        assert np.allclose(out[0, :5], 10.0)
        assert np.allclose(out[1, :5], 20.0)

    def test_constant_signal_all_equal(self):
        w = _windows(np.full((3, 128), 2.5), np.full((3, 128), 2.5))
        out = extract_ecdf_features(w)[0]
        assert np.allclose(out, 2.5)

    def test_shape(self):
        w = np.random.default_rng(0).normal(size=(7, 9, 128))
        assert extract_ecdf_features(w).shape == (7, 30)


class TestBands:
    def test_names_count(self):
        assert len(FEATURE_NAMES_BANDS) == 3 * len(BANDS_HZ) == 9

    def test_sine_energy_in_band(self):
        # 1.5Hz 全幅正弦（50Hz 取樣）：1-3Hz 帶占比應近乎 1
        t = np.arange(128) / 50.0
        sig = np.sin(2 * np.pi * 1.5 * t)
        w = _windows(np.zeros((3, 128)), np.stack([sig, sig * 0, sig * 0]))
        out = extract_band_features(w)[0]
        assert out[1] > 0.95          # total_x 1-3Hz
        assert out[0] < 0.05          # 0.3-1Hz
        assert out[2] < 0.05          # 3-6Hz

    def test_ratios_sum_leq_one(self):
        w = np.random.default_rng(1).normal(size=(5, 9, 128)) + 0.2
        out = extract_band_features(w)
        per_axis = out.reshape(5, 3, len(BANDS_HZ)).sum(axis=-1)
        assert np.all(per_axis <= 1.0 + 1e-6)

    def test_shape(self):
        w = np.random.default_rng(2).normal(size=(6, 9, 128))
        assert extract_band_features(w).shape == (6, 9)
