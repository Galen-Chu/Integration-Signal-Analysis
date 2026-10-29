"""UCI HAR loader · 原始慣性訊號、官方 561 特徵、標籤與受試者編號.

所有函式走「資料夾路徑」參數——測試可以用 tmp_path 造的迷你 fixture，
正式資料由 config.uci_root() 提供。
"""
import numpy as np

from src.isa.config import ACTIVITY_LABELS

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _read_floats(path) -> np.ndarray:
    return np.loadtxt(path)


def load_raw_signals(uci_root, split: str = "train") -> np.ndarray:
    """原始慣性訊號 → (N, 9, 128)；軸順序同 SIGNALS."""
    base = uci_root / split / "Inertial Signals"
    windows = []
    for name in SIGNALS:
        arr = _read_floats(base / f"{name}_{split}.txt")
        windows.append(arr.astype(np.float32))
    stacked = np.stack(windows, axis=1)   # (N, 9, 128)
    assert stacked.ndim == 3, f"expected (N,9,128), got {stacked.shape}"
    return stacked


def load_labels(uci_root, split: str = "train") -> np.ndarray:
    y = _read_floats(uci_root / split / f"y_{split}.txt").astype(int).ravel()
    assert set(y) <= set(ACTIVITY_LABELS), f"unknown labels: {set(y)}"
    return y


def load_subjects(uci_root, split: str = "train") -> np.ndarray:
    return _read_floats(uci_root / split / f"subject_{split}.txt").astype(int).ravel()


def load_feature_names(uci_root) -> list:
    lines = (uci_root / "features.txt").read_text().splitlines()
    names = [line.split(" ", 1)[1].strip() for line in lines if line.strip()]
    assert len(names) == 561, f"expected 561 features, got {len(names)}"
    return names


def load_uci_features(uci_root, split: str = "train") -> np.ndarray:
    """官方 561 維特徵矩陣（對照組：我們自製特徵 vs 官方工程特徵）."""
    return _read_floats(uci_root / split / f"X_{split}.txt")
