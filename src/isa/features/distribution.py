"""Distribution-shape features · ECDF 百分位數＋頻帶能量擴展（N2 移植）.

對照實驗定位：自製 49 維（ours2）與官方 561 維的最後差距集中在
靜態姿勢的「分佈形狀」資訊——官方集的 ECDF 百分位與 bandsEnergy
類特徵對 SITTING↔STANDING 有區別力。本模組以最小子集移植：

- ecdf 家族：body/total 六訊號 × [5,25,50,75,95] 百分位（分佈形狀）
  ——body 的中位數/四分位捕捉動作幅度對稱性，total 的捕捉重力帶穩定性
- bands 家族：total 三軸 × [0.3-1Hz（重力/靜態帶）, 1-3Hz（步頻帶）,
  3-6Hz（高頻動作帶）] 能量占比——靜態姿勢的訊息幾乎都在最低頻帶

全部向量化（numpy 軸運算＋批量 rfft），無逐視窗迴圈.
"""
import numpy as np
from scipy.fft import rfft

from src.isa.config import SAMPLING_RATE

ECDF_QUANTILES = (5, 25, 50, 75, 95)
BANDS_HZ = ((0.3, 1.0), (1.0, 3.0), (3.0, 6.0))

FEATURE_NAMES_ECDF = [
    f"{sig}_p{q}"
    for sig in ["body_x", "body_y", "body_z", "total_x", "total_y", "total_z"]
    for q in ECDF_QUANTILES
]
FEATURE_NAMES_BANDS = [
    f"total_{axis}_band{lo:g}_{hi:g}hz_ratio"
    for axis in "xyz"
    for lo, hi in BANDS_HZ
]


def extract_ecdf_features(windows: np.ndarray) -> np.ndarray:
    """windows: (N, 9, 128) → (N, 30)：六訊號 × 5 百分位（ECDF 反函數值）."""
    sigs = np.concatenate([windows[:, 0:3, :], windows[:, 6:9, :]], axis=1)  # (N,6,128)
    feats = np.percentile(sigs, ECDF_QUANTILES, axis=-1)     # (Q, N, 6)
    out = feats.transpose(1, 2, 0)                           # (N, 6, Q)
    return out.reshape(len(windows), -1).astype(np.float32)


def extract_band_features(windows: np.ndarray) -> np.ndarray:
    """windows: (N, 9, 128) → (N, 9)：total 三軸 × 三頻帶能量占比."""
    total = windows[:, 6:9, :]                                # (N,3,128)
    spectrum = np.abs(rfft(total, axis=-1)) ** 2              # (N,3,65)
    freqs = np.fft.rfftfreq(total.shape[-1], d=1 / SAMPLING_RATE)
    total_energy = spectrum.sum(axis=-1, keepdims=True)
    total_energy = np.where(total_energy > 0, total_energy, 1.0)
    cols = [((freqs >= lo) & (freqs < hi)).nonzero()[0] for lo, hi in BANDS_HZ]
    ratios = np.stack([spectrum[:, :, c].sum(axis=-1) / total_energy.squeeze(-1)
                       for c in cols], axis=-1)               # (N,3,B)
    return ratios.reshape(len(windows), -1).astype(np.float32)  # (N, 9)
