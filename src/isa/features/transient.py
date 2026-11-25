"""Transient features · 姿態轉換暫態特徵（v2 實驗，findings 指出的方向）.

靜態混淆（SITTING↔STANDING）的殘餘來自「視窗內不變的靜態訊號」——
49 維已是精簡集最適（N2 負結果），但視窗**內的變化**（坐→站的軀幹轉動、
慢速擺動）完全沒有特徵捕捉。本模組補上最小的暫態集合：

- 重力序列 g(t)＝total−body（UCI 布局的精確恆等式），其時間變化編碼
  「這個視窗裡姿態有沒有動」
- body_acc 的 0.2-1Hz 低頻能量占比——姿態轉換是慢暫態，與 1-3Hz 步頻帶
  分開
- 前後半窗能量比（log）——轉換的不對稱性

共 5 維；成功條件在 scripts/run_transient_ladder.py 的 docstring.
windows 軸順序同 loader.SIGNALS：0-2 body_acc、6-8 total_acc，50Hz.
"""

import numpy as np
from scipy.fft import rfft

from src.isa.config import SAMPLING_RATE, WINDOW_SIZE

FEATURE_NAMES = [
    "gravity_mag_std",
    "gravity_angle_change_deg",
    "gravity_angular_speed_deg_s",
    "body_lowfreq_ratio_02_1hz",
    "body_energy_log_ratio_2nd_half",
]

_HEAD = SAMPLING_RATE // 2  # 0.5s = 25 樣本
_CENTER_GAP_SEC = WINDOW_SIZE / SAMPLING_RATE - 0.5  # 兩段中心間隔 2.06s


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _lowfreq_ratio(body: np.ndarray, lo: float = 0.2, hi: float = 1.0) -> float:
    """0.2-1Hz 帶內能量占比——三軸各自 rfft 的**功率相加**後取比.

    不用合成量 |body|：單軸正弦會被 |·| 半波整流（頻率倍增），是偽象；
    三軸功率相加則不受軸向與整流影響（多軸自然能量加權）.
    """
    power = np.abs(rfft(body, axis=-1)) ** 2  # (3, 65)
    power = power.sum(axis=0)
    freqs = np.fft.rfftfreq(body.shape[-1], d=1 / SAMPLING_RATE)
    total = float(power[1:].sum())  # 排除 DC
    if total <= 0:
        return 0.0
    band = float(power[(freqs >= lo) & (freqs < hi)].sum())
    return band / total


def extract_transient_features(windows: np.ndarray) -> np.ndarray:
    """windows: (N, 9, 128) → (N, 5)."""
    body = windows[:, 0:3, :].astype(np.float64)
    total = windows[:, 6:9, :].astype(np.float64)
    gravity = total - body  # 低頻重力分量的時間序列
    mag_b = np.linalg.norm(body, axis=1)  # (N, 128)

    rows = []
    for i in range(len(windows)):
        g = gravity[i]
        gmag = np.linalg.norm(g, axis=0)  # |g(t)| 時間序列 (128,)
        gmag_std = float(gmag.std())  # 時間變化（靜態≈0）
        g_head = g[:, :_HEAD].mean(axis=1)
        g_tail = g[:, -_HEAD:].mean(axis=1)
        ang = _angle_deg(g_head, g_tail)
        e1 = float((mag_b[i][: len(mag_b[i]) // 2] ** 2).sum())
        e2 = float((mag_b[i][len(mag_b[i]) // 2 :] ** 2).sum())
        rows.append(
            [
                gmag_std,
                ang,
                ang / _CENTER_GAP_SEC,
                _lowfreq_ratio(body[i]),
                float(np.log((e2 + 1e-9) / (e1 + 1e-9))),
            ]
        )
    return np.asarray(rows, dtype=np.float32)
