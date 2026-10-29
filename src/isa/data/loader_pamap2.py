"""PAMAP2 loader · 心率＋IMU 同步資料（整合分析主力）.

檔案格式：subject{1..9}{0,1}.dat，空白分隔；
col 1=timestamp(s) · col 2=activityID · col 3=heart rate(bpm) ·
col 4-19 手部 IMU（col4 溫度、col5-7 ±6g 加速度計 xyz）
官方缺值以 NaN 表示；activityID=0 為 transient（不分析）.
"""
import numpy as np
import pandas as pd

PAMAP2_ACTIVITIES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "nordic_walking",
    12: "ascending_stairs", 13: "descending_stairs",
    16: "vacuum_cleaning", 17: "ironing", 24: "rope_jumping",
}
TRANSIENT = 0

# 載入欄位（pandas 0-based）：timestamp, activity, hr, 手部溫度, 手部 acc xyz
# PAMAP2 官方文件為 1-based，此處已換算
USECOLS = [0, 1, 2, 3, 4, 5, 6]
COL_NAMES = ["timestamp", "activity", "hr", "hand_temp",
             "acc_x", "acc_y", "acc_z"]


def load_subject(path, resample_sec: float = 1.0) -> pd.DataFrame:
    """單一受試者 → 逐秒重採樣 DataFrame（HR 缺值前向填充）."""
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=USECOLS,
                     names=COL_NAMES, na_values="NaN")
    df = df[df["activity"] != TRANSIENT]
    if df.empty:
        return df
    df["t"] = np.floor(df["timestamp"] / resample_sec)
    agg = df.groupby("t").agg(
        activity=("activity", "first"),
        hr=("hr", "mean"),
        acc_mag=("acc_x", lambda s: float("nan")),  # placeholder，下面重算
    )
    # 加速度合成量：逐秒平均（先對原始列算 mag 再 groupby 平均，順序才正確）
    df["acc_mag_inst"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
    agg["acc_mag"] = df.groupby("t")["acc_mag_inst"].mean()
    agg["hr"] = agg["hr"].ffill()          # HR 官方建議前向填充
    agg = agg[agg["hr"].notna()]
    agg["activity"] = agg["activity"].astype(int)
    return agg.reset_index()


def load_all_protocol(data_dir) -> pd.DataFrame:
    """subject101-109.dat（Protocol）全部載入並附受試者編號."""
    frames = []
    for path in sorted(data_dir.glob("subject10*.dat")):
        subject_id = int(path.stem.replace("subject", ""))
        if subject_id > 109:      # optional sessions (subjectxx1.dat) 先不用
            continue
        df = load_subject(path)
        if df.empty:
            continue
        df["subject"] = subject_id
        frames.append(df)
        print(f"  {path.name}: {len(df)} seconds, "
              f"{df['activity'].nunique()} activities")
    if not frames:
        raise FileNotFoundError(
            f"no subject10*.dat under {data_dir} — run fetch_data first")
    return pd.concat(frames, ignore_index=True)
