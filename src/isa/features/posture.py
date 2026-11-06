"""Posture-geometry features · 姿態幾何特徵（v2 實驗）.

對照實驗證實：靜態姿勢（SITTING↔STANDING）混淆主因是精簡特徵缺少
重力/角度資訊（官方 561 維把 SIT 召回 55%→90%）。本模組補上最小集合：
重力向量（total−body 的視窗平均，DC 分量）與各軸對重力的夾角——
共 6 維，43+6=49 維仍遠小於 561，目標以 <50 維逼近官方表現.

windows 軸順序同 loader.SIGNALS：0-2 body_acc、6-8 total_acc.
"""
import numpy as np

FEATURE_NAMES = [
    "gravity_x", "gravity_y", "gravity_z",
    "angle_x_gravity_deg", "angle_y_gravity_deg", "angle_z_gravity_deg",
]


def gravity_vector(window: np.ndarray) -> np.ndarray:
    """單一視窗的重力向量＝mean(total_acc − body_acc)，shape (3,).

    body_acc 已除重力（均值≈0），total_acc 含重力——差的均值即 DC 重力.
    """
    return (window[6:9, :] - window[0:3, :]).mean(axis=1)


def axis_gravity_angles(g: np.ndarray) -> list:
    """各軸與重力向量的夾角（度）；|g|≈0（失重/缺值）時回 90°."""
    norm = float(np.linalg.norm(g))
    if norm < 1e-9:
        return [90.0, 90.0, 90.0]
    return [float(np.degrees(np.arccos(np.clip(c / norm, -1.0, 1.0))))
            for c in g]


def extract_posture_features(windows: np.ndarray) -> np.ndarray:
    """windows: (N, 9, 128) → (N, 6)：重力 3 維＋夾角 3 維."""
    rows = []
    for i in range(len(windows)):
        g = gravity_vector(windows[i])
        rows.append([float(c) for c in g] + axis_gravity_angles(g))
    return np.asarray(rows, dtype=np.float32)
