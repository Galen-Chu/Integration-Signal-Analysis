"""Physiological analysis · Fitbit 心率：強度分區、恢復率、每日負荷.

實機/臨床版可換 RR-interval 做 HRV；Fitbit 秒級心率資料先做
zone 分布與恢復率（HRR）——動作「負荷」的量化基礎.
"""
import numpy as np
import pandas as pd

from src.isa.config import DEFAULT_MAX_HR, HR_ZONES


def hr_zone_percentages(hr_series: pd.Series, max_hr: int = DEFAULT_MAX_HR) -> dict:
    """各強度區時間占比（%HRmax 分區）."""
    pct = hr_series / max_hr * 100
    total = len(pct)
    result = {}
    for name, lo, hi in HR_ZONES:
        result[name] = float(((pct >= lo) & (pct < hi)).sum() / total * 100)
    result["_mean_pct_hrmax"] = float(pct.mean())
    return result


def hr_recovery(hr: pd.Series, times: pd.Series,
                anchor: pd.Timestamp, minutes: float = 1.0):
    """心率恢復（HRR）：anchor 當下峰值 − 之後 N 分鐘的最低心率.

    HRR < 12 bpm/1min 是疲勞/心血管風險的經典門檻.
    回傳 None 表示資料不足（視窗內無後續心率）.
    """
    mask_after = times > anchor
    baseline_mask = (times > anchor) & (times <= anchor + pd.Timedelta(minutes=minutes))
    if not baseline_mask.any():
        return None
    peak = float(hr[times <= anchor].tail(60).max())  # anchor 前一分鐘峰值
    later_min = float(hr[baseline_mask].min())
    return round(peak - later_min, 1)


def daily_load_summary(minute_hr: pd.Series) -> dict:
    """每日負荷：分鐘級心率的時數×強度加權（簡化 TRIMP）."""
    pct = minute_hr / DEFAULT_MAX_HR * 100
    weights = np.select(
        [pct < 50, pct < 60, pct < 70, pct < 85],
        [0, 1, 2, 4], default=8)          # 越高強度權重越高
    return {
        "weighted_load_minutes": float(weights.sum()),
        "moderate_plus_minutes": float((pct >= 60).sum()),
        "peak_hr": int(minute_hr.max()),
        "resting_hr_proxy": int(minute_hr.quantile(0.05)),
    }


def load_fitbit_heartrate(fitbit_dir, file_glob: str = "heartrate_seconds*") -> pd.DataFrame:
    """載入 Fitbit 秒級心率 CSV（每位使用者一檔）→ 單一 DataFrame."""
    frames = []
    for path in sorted(fitbit_dir.rglob(f"{file_glob}.csv")):
        df = pd.read_csv(path)
        # 慣例欄位：Id, Time, Value
        df.columns = [c.strip() for c in df.columns]
        df["Time"] = pd.to_datetime(df["Time"])
        df["Id"] = df["Id"].astype(int)
        frames.append(df[["Id", "Time", "Value"]])
    if not frames:
        raise FileNotFoundError(
            f"no '{file_glob}.csv' under {fitbit_dir} — run scripts/fetch_data.py")
    return pd.concat(frames, ignore_index=True).rename(columns={"Value": "hr"})
