"""run_pamap2_classification · PAMAP2 分類複現（item-2 實驗）.

    python scripts/run_pamap2_classification.py
問題：UCI 的 SITTING↔STANDING 混淆（與「特徵假說」）會在第二個資料集重現嗎？
12 活動（含家務）、9 受試者、與 UCI 相同的 49 維特徵與 GroupKFold 協議.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.isa.config import FIGURES_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402
from src.isa.data.loader_pamap2 import (PAMAP2_ACTIVITIES,  # noqa: E402
                                        load_pamap2_windows)
from src.isa.features.freq_domain import extract_freq_features  # noqa: E402
from src.isa.features.posture import extract_posture_features  # noqa: E402
from src.isa.features.time_domain import extract_time_features  # noqa: E402
from src.isa.models.train import subject_cv_evaluate  # noqa: E402
from src.isa.viz.plots import plot_confusion  # noqa: E402


def main():
    data_dir = RAW_DIR / "PAMAP2_Dataset" / "Protocol"
    if not data_dir.is_dir():
        raise SystemExit("PAMAP2 未下載——見 README（download_pamap2()）")

    print("Building UCI-compatible windows from PAMAP2 Protocol…")
    windows, y, groups = load_pamap2_windows(data_dir)
    counts = {PAMAP2_ACTIVITIES[k]: int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"windows: {windows.shape}, subjects: {sorted(np.unique(groups).tolist())}")
    print("windows/activity:", counts)

    X = np.hstack([extract_time_features(windows),
                   extract_freq_features(windows),
                   extract_posture_features(windows)])
    print(f"features: {X.shape[1]}-dim (ours2)")

    reports = {}
    for kind in ("rf", "svm"):
        print(f"\n=== {kind.upper()} · PAMAP2 subject-wise 5-fold CV ===")
        report = subject_cv_evaluate(X, y, groups, kind=kind)
        reports[kind] = report
        print(f"mean acc = {report['mean_accuracy']:.4f} "
              f"(±{report['std_accuracy']:.4f}), macro-F1 = {report['macro_f1']:.4f}")

    # 靜態對（sitting/standing/lying）召回——複現檢驗的核心
    labels = reports["rf"]["labels"]
    names = [PAMAP2_ACTIVITIES[int(l)] for l in labels]
    for kind in ("rf", "svm"):
        cm = np.array(reports[kind]["confusion_matrix"])
        recall = cm.diagonal() / cm.sum(axis=1)
        print(f"\n{kind.upper()} 靜態召回：")
        for i, name in enumerate(names):
            if name in ("lying", "sitting", "standing"):
                print(f"  {name:10s} recall={recall[i]:.2f}")
        plot_confusion(cm, labels, names,
                       FIGURES_DIR / f"confusion_{kind}_pamap2_ours2.png")

    out = PROCESSED_DIR / "report_pamap2_ours2.json"
    out.write_text(json.dumps(reports, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
