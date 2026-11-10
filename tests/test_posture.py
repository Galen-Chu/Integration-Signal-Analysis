"""姿態幾何特徵測試——已知重力的合成視窗、精確角度."""
import numpy as np
import pytest

from src.isa.features.posture import axis_gravity_angles, extract_posture_features, gravity_vector


def _window(gravity, body_noise=0.0, n=128, seed=0):
    """造 (9,128)：total = body + gravity（DC）；body≈噪訊."""
    rng = np.random.default_rng(seed)
    w = np.zeros((9, n))
    g = np.asarray(gravity, dtype=float)
    w[0:3, :] = body_noise * rng.normal(size=(3, n))   # body_acc
    w[6:9, :] = w[0:3, :] + g[:, None]                 # total_acc = body + g
    return w


class TestGravityVector:
    def test_pure_gravity_extraction(self):
        g = gravity_vector(_window([0.2, 0.1, 0.97]))
        assert np.allclose(g, [0.2, 0.1, 0.97], atol=1e-6)

    def test_body_noise_does_not_biase(self):
        # body 均值應被平均掉（噪訊均值→0）
        g = gravity_vector(_window([0.0, 0.0, 1.0], body_noise=0.5))
        assert np.allclose(g, [0.0, 0.0, 1.0], atol=0.05)


class TestAngles:
    def test_gravity_along_z(self):
        # 手機平放：重力全在 Z → Z 夾角 0°，X/Y 90°
        assert axis_gravity_angles(np.array([0.0, 0.0, 1.0])) == pytest.approx([90.0, 90.0, 0.0])

    def test_tilted_45(self):
        # 重力在 X-Z 平面 45° → X=45°, Z=45°, Y=90°
        g = np.array([0.7071, 0.0, 0.7071])
        assert axis_gravity_angles(g) == pytest.approx([45.0, 90.0, 45.0], abs=0.1)

    def test_zero_gravity_returns_90(self):
        assert axis_gravity_angles(np.zeros(3)) == [90.0, 90.0, 90.0]


class TestExtract:
    def test_shape_and_sit_stand_separation(self):
        # 坐姿（直立，重力在 Z）vs 躺姿（重力在 Y）——姿態特徵應截然不同
        sit = _window([0.0, 0.1, 0.99], body_noise=0.3, seed=1)
        lie = _window([0.0, 0.99, 0.1], body_noise=0.3, seed=2)
        feats = extract_posture_features(np.stack([sit, lie]))
        assert feats.shape == (2, 6)
        # 坐姿 Z 夾角遠小於躺姿的 Z 夾角
        assert feats[0, 5] < 30 < feats[1, 5]
        # 重力分量符號正確
        assert feats[1, 1] > 0.9
