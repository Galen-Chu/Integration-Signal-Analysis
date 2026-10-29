"""Time-domain features · 時域特徵（每視窗 128 樣本×9 訊號）."""
import numpy as np

FEATURE_NAMES = [
    *[f"{sig}_{stat}" for sig in ["acc_x", "acc_y", "acc_z", "acc_mag"]
      for stat in ["mean", "std", "mad", "range", "energy", "sma"]],
    "acc_xy_corr", "acc_xz_corr", "acc_yz_corr",
]


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation（穩健離散度量）."""
    return float(np.median(np.abs(x - np.median(x))))


def _sma_axis(x: np.ndarray) -> float:
    """Signal magnitude area（單軸）：|x| 積分／樣本數."""
    return float(np.sum(np.abs(x)) / len(x))


def sma_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """三軸 SMA（UCI 定義：三軸 |·| 之和除以樣本數）."""
    return float((np.sum(np.abs(x)) + np.sum(np.abs(y)) + np.sum(np.abs(z)))
                 / len(x))


def extract_time_features(windows: np.ndarray) -> np.ndarray:
    """windows: (N, 9, 128)；取 total_acc 前三軸（0-2 為 body_acc）計算.

    回傳 (N, len(FEATURE_NAMES))。使用 body_acc（已去重力的身體加速度）
    ——動作辨識的主力訊號；軸 0/1/2 = body_acc x/y/z。
    """
    body = windows[:, 0:3, :]                       # (N, 3, 128)
    mag = np.sqrt((body ** 2).sum(axis=1))          # (N, 128) 三軸合成量

    rows = []
    for i in range(len(windows)):
        bx, by, bz = body[i]
        m = mag[i]
        per_signal = []
        for sig in (bx, by, bz, m):
            per_signal += [
                float(sig.mean()), float(sig.std()), _mad(sig),
                float(sig.max() - sig.min()),
                float(np.sum(sig ** 2) / len(sig)),
                _sma_axis(sig),
            ]
        corr = [float(np.corrcoef(a, b)[0, 1])
                for a, b in ((bx, by), (bx, bz), (by, bz))]
        rows.append(per_signal + corr)
    return np.asarray(rows, dtype=np.float32)
