"""Frequency-domain features · FFT 頻域特徵."""
import numpy as np
from scipy.fft import rfft

from src.isa.config import SAMPLING_RATE

# 命名順序必須與 extract_freq_features 的提取順序一致：
# 每個訊號依序產生 domfreq/centroid/band/entropy 四欄（交錯，非逐特徵分組）
FEATURE_NAMES = [
    f"{sig}_{feat}" for sig in ["acc_x", "acc_y", "acc_z", "acc_mag"]
    for feat in ["domfreq_hz", "centroid_hz", "band1_3hz_ratio",
                 "spectral_entropy"]
]


def dominant_frequency_hz(x: np.ndarray, fs: int = SAMPLING_RATE) -> float:
    """主頻（去除 DC 後振幅最大的頻率）."""
    spectrum = np.abs(rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    return float(freqs[1:][np.argmax(spectrum[1:])])


def spectral_centroid_hz(x: np.ndarray, fs: int = SAMPLING_RATE) -> float:
    """頻譜質心：能量加權平均頻率（動作節奏快慢的指紋）.

    全零訊號（凍結軸去重力後的殘差）定義為 0——0/0 無定義且下游模型拒收 NaN.
    """
    spectrum = np.abs(rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    power = spectrum ** 2
    total = power.sum()
    return float((freqs * power).sum() / total) if total > 0 else 0.0


def band_energy_ratio(x: np.ndarray, lo: float, hi: float,
                      fs: int = SAMPLING_RATE) -> float:
    """特定頻帶能量占比（步行步頻約落在 1-3Hz）."""
    spectrum = np.abs(rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    total = spectrum.sum()
    band = spectrum[(freqs >= lo) & (freqs < hi)].sum()
    return float(band / total) if total > 0 else 0.0


def spectral_entropy(x: np.ndarray, fs: int = SAMPLING_RATE) -> float:
    """頻譜熵（常態化 0-1）：規律動作低熵、混亂訊號高熵."""
    spectrum = np.abs(rfft(x)) ** 2
    p = spectrum / spectrum.sum() if spectrum.sum() > 0 else spectrum
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / np.log2(len(spectrum)))


def extract_freq_features(windows: np.ndarray) -> np.ndarray:
    """windows: (N, 9, 128) body_acc 三軸＋合成量 → (N, 16)."""
    body = windows[:, 0:3, :]
    mag = np.sqrt((body ** 2).sum(axis=1))
    rows = []
    for i in range(len(windows)):
        per = []
        for sig in (body[i, 0], body[i, 1], body[i, 2], mag[i]):
            per += [
                dominant_frequency_hz(sig),
                spectral_centroid_hz(sig),
                band_energy_ratio(sig, 1.0, 3.0),
                spectral_entropy(sig),
            ]
        rows.append(per)
    return np.asarray(rows, dtype=np.float32)


def combine(*feature_matrices) -> np.ndarray:
    return np.hstack([np.asarray(m, dtype=np.float32) for m in feature_matrices])
