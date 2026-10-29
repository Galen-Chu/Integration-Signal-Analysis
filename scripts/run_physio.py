"""run_physio · Fitbit 心率生理分析（I4）：分區、負荷、恢復力.

    python scripts/run_physio.py
產出：
    reports/figures/hr_timeline_*.png      代表使用者心率時間軸
    reports/figures/hr_recovery_dist.png  每日峰值後 HRR 分布
    data/processed/physio_summary.json    全體彙整（分區/負荷/HRR）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.isa.config import FIGURES_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402
from src.isa.physio.hr_analysis import (daily_load_summary,  # noqa: E402
                                        hr_recovery,
                                        hr_zone_percentages)
from src.isa.viz.plots import plot_hr_timeline  # noqa: E402

FITBIT_DIR = RAW_DIR / "fitbit"


def load_minute_hr(fitbit_dir: Path) -> pd.DataFrame:
    """分鐘級心率（heartrate_minute 或由秒級重採樣）."""
    frames = []
    for pattern in ("heartrate_minute*", "heartrate_seconds*"):
        for path in sorted(fitbit_dir.rglob(f"{pattern}.csv")):
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]
            if "ActivityMinute" in df.columns:      # minute 版欄位
                time_col, hr_col = "ActivityMinute", "Calories"  # placeholder
                continue
            df["Time"] = pd.to_datetime(df["Time"])
            frames.append(df[["Id", "Time", "Value"]])
    if not frames:
        raise FileNotFoundError("找不到 Fitbit 心率 CSV——先跑 fetch_data.py --fitbit")
    df = pd.concat(frames, ignore_index=True).rename(columns={"Value": "hr"})
    # 秒級 → 分鐘級（每分鐘平均）
    df["minute"] = df["Time"].dt.floor("min")
    return (df.groupby(["Id", "minute"])["hr"].mean()
              .reset_index().rename(columns={"minute": "Time"}))


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    hr = load_minute_hr(FITBIT_DIR)
    print(f"loaded {len(hr):,} user-minutes from {hr['Id'].nunique()} users")

    per_user, hrr_values = [], []
    top_users = (hr.groupby("Id")["hr"].count()
                   .sort_values(ascending=False).index[:3].tolist())

    for uid, grp in hr.groupby("Id"):
        zones = hr_zone_percentages(grp["hr"])
        load = daily_load_summary(grp["hr"])
        per_user.append({"Id": int(uid), **zones, **load})

        # 每日峰值後的 HRR（取每天最大心率時點，往後 1 分鐘）
        for day, day_hr in grp.groupby(grp["Time"].dt.date):
            anchor_idx = day_hr["hr"].idxmax()
            anchor = day_hr.loc[anchor_idx]
            hrr = hr_recovery(grp["hr"], grp["Time"], anchor=anchor["Time"])
            if hrr is not None:
                hrr_values.append(hrr)

        if uid in top_users:
            day0 = grp["Time"].dt.date.value_counts().index[0]  # 資料最完整的一天
            one_day = grp[grp["Time"].dt.date == day0]
            plot_hr_timeline(one_day["hr"].to_numpy(), one_day["Time"],
                             FIGURES_DIR / f"hr_timeline_{int(uid)}.png",
                             title=f"User {int(uid)} · {day0} (minute-level HR)")
            print(f"  timeline figure: user {int(uid)} {day0}")

    hrr = np.array(hrr_values)
    print(f"\nHRR after daily peaks: n={len(hrr)}, "
          f"median={np.median(hrr):.1f} bpm, IQR=({np.percentile(hrr, 25):.1f}, "
          f"{np.percentile(hrr, 75):.1f}), <12bpm 比例={float((hrr < 12).mean()) * 100:.1f}%")

    # HRR 分布圖
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(hrr, bins=40, color="#2a78d6", edgecolor="white", linewidth=0.5)
    ax.axvline(12, color="#d03b3b", lw=1.5, ls="--")
    ax.text(12.5, ax.get_ylim()[1] * 0.9, "HRR=12 bpm\n(風險門檻)", fontsize=9,
            color="#d03b3b")
    ax.set_xlabel("Heart-rate recovery after daily peak (bpm)")
    ax.set_ylabel("days")
    ax.set_title("HRR distribution — all users, after each day's peak effort")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "hr_recovery_dist.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_users": int(hr["Id"].nunique()),
        "n_user_minutes": int(len(hr)),
        "zone_means": {k: round(float(np.mean([u[k] for u in per_user])), 2)
                       for k in ("rest", "light", "moderate", "vigorous", "near_max")},
        "hrr_after_daily_peaks": {
            "n": int(len(hrr)),
            "median_bpm": float(np.median(hrr)),
            "p25": float(np.percentile(hrr, 25)),
            "p75": float(np.percentile(hrr, 75)),
            "pct_below_12bpm": float((hrr < 12).mean() * 100),
        },
        "per_user": per_user,
    }
    out = PROCESSED_DIR / "physio_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"summary → {out}")


if __name__ == "__main__":
    main()
