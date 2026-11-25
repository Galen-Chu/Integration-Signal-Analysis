"""暫態特徵測試——合成訊號精確值（靜態≈0、轉態大角度、頻帶分離）."""

import numpy as np

from src.isa.features.transient import (
    FEATURE_NAMES,
    extract_transient_features,
)

FS = 50
T = np.arange(128) / FS


def _window(body, gravity):
    """body (3,128)＋gravity (3,128) → UCI 布局視窗：total=body+gravity."""
    w = np.zeros((9, 128), dtype=np.float32)
    w[0:3, :] = body
    w[6:9, :] = body + gravity
    return w


def _static(gravity=(0.0, 0.0, 9.81), noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    body = rng.normal(0, noise, (3, 128))
    return _window(body, np.array(gravity)[:, None] * np.ones(128))


class TestTransient:
    def test_shape_and_names(self):
        rng = np.random.default_rng(1)
        windows = rng.normal(size=(4, 9, 128)).astype(np.float32)
        feats = extract_transient_features(windows)
        assert feats.shape == (4, 5)
        assert len(FEATURE_NAMES) == 5
        assert np.isfinite(feats).all()

    def test_static_window_is_quiet(self):
        """靜態視窗：重力不動、無低頻動作 → 角度變化≈0、重力 std≈0."""
        f = extract_transient_features(_static()[None])[0]
        assert abs(f[1]) < 2.0  # gravity_angle_change_deg
        assert abs(f[2]) < 1.0  # deg/s
        assert f[0] < 0.05  # gravity_mag_std

    def test_posture_change_detected(self):
        """視窗中段重力翻轉（z→x 約 90°）→ 大角度變化."""
        w = _static()[None].copy()
        w2 = _window(np.zeros((3, 128)), np.array([9.81, 0, 0])[:, None] * np.ones(128))[None]
        mixed = w.copy()
        mixed[0][6:9, 64:] = w2[0][6:9, 64:]  # 只換後半：z 重力 → x 重力
        f = extract_transient_features(mixed)[0]
        assert f[1] > 45.0, f"angle change {f[1]:.1f} deg"
        assert f[2] > 20.0

    def test_lowfreq_band_separation(self):
        """0.78Hz 慢擺（窗上恰 2 週期、落 0.2-1Hz 帶內）vs 5Hz 快振動."""
        slow = np.sin(2 * np.pi * 0.78125 * T) * 0.5  # bin 頻率：乾淨落帶內
        fast = np.sin(2 * np.pi * 5.0 * T) * 0.5
        w_slow = _window(
            np.vstack([slow, np.zeros(128), np.zeros(128)]),
            np.array([0, 0, 9.81])[:, None] * np.ones(128),
        )[None]
        w_fast = _window(
            np.vstack([fast, np.zeros(128), np.zeros(128)]),
            np.array([0, 0, 9.81])[:, None] * np.ones(128),
        )[None]
        f = extract_transient_features(np.r_[w_slow, w_fast])
        assert f[0, 3] > 0.5  # 慢擺能量集中低頻帶
        assert f[1, 3] < 0.05  # 快振動幾乎不在 0.2-1Hz
        assert f[0, 3] > 10 * max(f[1, 3], 1e-4)

    def test_energy_asymmetry_sign(self):
        """前強後弱 → log 比為負；前弱後強 → 為正."""
        env1 = np.exp(-T / 0.4)  # 衰減
        env2 = np.exp(-(T.max() - T) / 0.4)  # 反向（增強）
        grow = np.sin(2 * np.pi * 3 * T) * env1 * 2
        decay = np.sin(2 * np.pi * 3 * T) * env2 * 2
        w1 = _window(
            np.vstack([grow, np.zeros(128), np.zeros(128)]),
            np.array([0, 0, 9.81])[:, None] * np.ones(128),
        )[None]
        w2 = _window(
            np.vstack([decay, np.zeros(128), np.zeros(128)]),
            np.array([0, 0, 9.81])[:, None] * np.ones(128),
        )[None]
        f = extract_transient_features(np.r_[w1, w2])
        assert f[0, 4] < 0 < f[1, 4]
