"""run_integration · PAMAP2 動作×心率負荷整合（I4 旗艦分析）.

    python scripts/run_integration.py
產出：
    reports/figures/activity_hr_load_ladder.png  各活動平均 %HRmax 排序梯
    data/processed/activity_hr_load.json         每活動 HR/加速度統計
問題：同一群人做不同動作時，身體（心率）到底付出多少負荷？
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.isa.config import (DEFAULT_MAX_HR, FIGURES_DIR,  # noqa: E402
                            PROCESSED_DIR, RAW_DIR)
from src.isa.data.loader_pamap2 import (PAMAP2_ACTIVITIES,  # noqa: E402
                                        load_all_protocol)


def main():
    data_dir = RAW_DIR / "PAMAP2_Dataset" / "Protocol"
    if not data_dir.is_dir():
        raise SystemExit("PAMAP2 未下載——見 README（download_pamap2()）")

    print("Loading PAMAP2 Protocol (9 subjects, 1-second resample)…")
    df = load_all_protocol(data_dir)
    print(f"total: {len(df):,} person-seconds, "
          f"{df['subject'].nunique()} subjects, "
          f"{df['activity'].nunique()} activities")

    # 每活動統計（跨受試者總合）：平均 %HRmax、加速度合成量、樣本數
    rows = []
    for act_id, grp in df.groupby("activity"):
        if act_id not in PAMAP2_ACTIVITIES:
            continue
        rows.append({
            "activity": PAMAP2_ACTIVITIES[act_id],
            "act_id": int(act_id),
            "pct_hrmax_mean": float((grp["hr"] / DEFAULT_MAX_HR * 100).mean()),
            "pct_hrmax_std": float((grp["hr"] / DEFAULT_MAX_HR * 100).std()),
            "hr_mean": float(grp["hr"].mean()),
            "acc_mag_mean": float(grp["acc_mag"].mean()),
            "n_subjects": int(grp["subject"].nunique()),
            "n_seconds": int(len(grp)),
        })
    rows.sort(key=lambda r: r["pct_hrmax_mean"])
    for r in rows:
        print(f"  {r['activity']:20s} {r['pct_hrmax_mean']:5.1f} %HRmax "
              f"(±{r['pct_hrmax_std']:4.1f})  hr={r['hr_mean']:5.1f}bpm  "
              f"acc={r["acc_mag_mean"]:5.1f}m/s²  n={r['n_seconds']}s")

    # 旗艦圖：活動負荷梯（%HRmax 排序長條，dataviz：≤24px、4px 圓角端、端點值）
    names = [r["activity"] for r in rows]
    vals = [r["pct_hrmax_mean"] for r in rows]
    errs = [r["pct_hrmax_std"] for r in rows]
    colors = ["#2a78d6" if v < 60 else "#eb6834" if v < 70 else "#1baf7a"
              for v in vals]  # light/moderate/vigorous（zone 對應色，圖例說明）

    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(rows))
    ax.barh(y, vals, xerr=errs, height=0.62, color=colors,
            error_kw=dict(ecolor="#c3c2b7", lw=1), zorder=3)
    ax.set_yticks(y, labels=names)
    for i, v in enumerate(vals):
        ax.text(v + errs[i] + 1, i, f"{v:.0f}%", va="center",
                fontsize=9, color="#0b0b0b")
    for zone, xv in (("light 60%", 60), ("moderate 70%", 70), ("vigorous 85%", 85)):
        ax.axvline(xv, color="#c3c2b7", lw=1, ls="--", zorder=1)
        ax.text(xv, len(rows) - 0.3, zone, fontsize=8, color="#898781",
                ha="center")
    ax.set_xlabel("Mean %HRmax (±1 SD across person-seconds)")
    ax.set_title("Activity load ladder — what each movement costs the heart\n"
                 f"PAMAP2, {df['subject'].nunique()} subjects, HRmax assumed "
                 f"{DEFAULT_MAX_HR}bpm")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color="#e1e0d9", lw=1, zorder=0)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "activity_hr_load_ladder.png", dpi=150)
    plt.close(fig)

    out = PROCESSED_DIR / "activity_hr_load.json"
    out.write_text(json.dumps(
        {"subjects": int(df["subject"].nunique()),
         "total_person_seconds": int(len(df)),
         "hrmax_assumed": DEFAULT_MAX_HR,
         "activities": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfigure → {FIGURES_DIR / 'activity_hr_load_ladder.png'}")
    print(f"summary → {out}")


if __name__ == "__main__":
    main()
