"""run_pipeline · 端到端管線：UCI HAR → 自製特徵 → RF/SVM → subject-wise 評估.

    python scripts/run_pipeline.py                # 自製特徵（時域+頻域）
    python scripts/run_pipeline.py --features uci # 官方 561 特徵（對照組）
    python scripts/run_pipeline.py --features both
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.isa.config import FIGURES_DIR, PROCESSED_DIR, UCI_DIR  # noqa: E402
from src.isa.data.loader import (load_labels, load_raw_signals,  # noqa: E402
                                 load_subjects, load_uci_features)
from src.isa.features.freq_domain import extract_freq_features  # noqa: E402
from src.isa.features.time_domain import extract_time_features  # noqa: E402
from src.isa.models.train import subject_cv_evaluate  # noqa: E402
from src.isa.viz.plots import plot_confusion  # noqa: E402


def build_features(split: str, kind: str) -> np.ndarray:
    if kind in ("ours", "both"):
        windows = load_raw_signals(UCI_DIR, split)
        print(f"[{split}] raw windows: {windows.shape} → extracting features…")
        ours = np.hstack([extract_time_features(windows),
                          extract_freq_features(windows)])
    if kind == "ours":
        return ours
    official = load_uci_features(UCI_DIR, split)
    if kind == "uci":
        return official
    return np.hstack([ours, official])


def main():
    parser = argparse.ArgumentParser(description="HAR 管線：特徵→模型→評估")
    parser.add_argument("--features", choices=["ours", "uci", "both"],
                        default="ours",
                        help="特徵來源（ours=自製時域+頻域；uci=官方561；both=併）")
    parser.add_argument("--models", nargs="+", default=["rf", "svm"])
    args = parser.parse_args()

    if not (UCI_DIR / "train").is_dir():
        raise SystemExit("UCI HAR 未下載——先跑 python scripts/fetch_data.py --uci")

    X = build_features("train", args.features)
    y = load_labels(UCI_DIR, "train")
    groups = load_subjects(UCI_DIR, "train")
    print(f"X: {X.shape}, subjects: {len(np.unique(groups))}, "
          f"windows/label: {np.bincount(y)[1:].tolist()}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    reports = {}
    for kind in args.models:
        print(f"\n=== {kind.upper()} · subject-wise 5-fold CV ===")
        report = subject_cv_evaluate(X, y, groups, kind=kind)
        reports[kind] = report
        print(f"mean acc = {report['mean_accuracy']:.4f} "
              f"(±{report['std_accuracy']:.4f}), macro-F1 = {report['macro_f1']:.4f}")

    out = PROCESSED_DIR / f"cv_report_{args.features}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    from src.isa.config import ACTIVITY_LABELS
    labels = reports[args.models[0]]["labels"]
    names = [ACTIVITY_LABELS[int(l)] for l in labels]
    for kind in args.models:
        cm = np.array(reports[kind]["confusion_matrix"])
        plot_confusion(cm, labels, names,
                       FIGURES_DIR / f"confusion_{kind}_{args.features}.png")
    out.write_text(json.dumps(reports, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nreport → {out}\nfigures → {FIGURES_DIR}")


if __name__ == "__main__":
    main()
