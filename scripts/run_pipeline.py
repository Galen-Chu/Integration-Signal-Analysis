"""run_pipeline · 端到端管線：UCI HAR → 特徵 → RF/SVM → 誠實評估.

    python scripts/run_pipeline.py                          # 自製特徵 × GroupKFold 5 折
    python scripts/run_pipeline.py --features uci           # 官方 561 特徵（對照組）
    python scripts/run_pipeline.py --features both          # 併集
    python scripts/run_pipeline.py --split official         # 官方 test split（21訓練→9測試）
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.isa.config import FIGURES_DIR, PROCESSED_DIR, UCI_DIR  # noqa: E402
from src.isa.data.loader import (  # noqa: E402
    load_labels,
    load_raw_signals,
    load_subjects,
    load_uci_features,
)
from src.isa.features.freq_domain import extract_freq_features  # noqa: E402
from src.isa.features.time_domain import extract_time_features  # noqa: E402
from src.isa.models.train import evaluate_holdout, subject_cv_evaluate  # noqa: E402
from src.isa.viz.plots import plot_confusion  # noqa: E402


def build_features(split: str, kind: str) -> np.ndarray:
    from src.isa.features.posture import extract_posture_features
    if kind in ("ours", "ours2", "both"):
        windows = load_raw_signals(UCI_DIR, split)
        print(f"[{split}] raw windows: {windows.shape} → extracting features…")
        ours = np.hstack([extract_time_features(windows),
                          extract_freq_features(windows)])
    if kind == "ours":
        return ours
    if kind == "ours2":
        return np.hstack([ours, extract_posture_features(windows)])
    official = load_uci_features(UCI_DIR, split)
    if kind == "uci":
        return official
    return np.hstack([ours, official])


def finish(reports: dict, args) -> None:
    """混淆矩陣圖＋報告存檔（CV 與 official 共用收尾）."""
    from src.isa.config import ACTIVITY_LABELS
    out = PROCESSED_DIR / f"report_{args.split}_{args.features}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = reports[args.models[0]]["labels"]
    names = [ACTIVITY_LABELS[int(code)] for code in labels]
    for kind in args.models:
        cm = np.array(reports[kind]["confusion_matrix"])
        plot_confusion(cm, labels, names,
                       FIGURES_DIR / f"confusion_{kind}_{args.split}_{args.features}.png")
    out.write_text(json.dumps(reports, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nreport → {out}\nfigures → {FIGURES_DIR}")


def main():
    parser = argparse.ArgumentParser(description="HAR 管線：特徵→模型→評估")
    parser.add_argument("--features", choices=["ours", "ours2", "uci", "both"],
                        default="ours",
                        help="ours=時域+頻域(43)；ours2=+姿態幾何(49)；"
                             "uci=官方561；both=ours+uci")
    parser.add_argument("--models", nargs="+", default=["rf", "svm"])
    parser.add_argument("--split", choices=["cv", "official"], default="cv",
                        help="cv=GroupKFold 5 折（訓練集內）；official=官方 test"
                             "（21 人訓練→9 人測試，與文獻同條件）")
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

    if args.split == "official":
        X_te = build_features("test", args.features)
        y_te = load_labels(UCI_DIR, "test")
        subjects_te = load_subjects(UCI_DIR, "test")
        print(f"official holdout: train {len(np.unique(groups))} subjects → "
              f"test {len(np.unique(subjects_te))} subjects "
              f"({len(y_te)} windows)")
        for kind in args.models:
            print(f"\n=== {kind.upper()} · official holdout ({args.features}) ===")
            report = evaluate_holdout(X, y, X_te, y_te, groups, kind=kind)
            reports[kind] = report
            print(f"accuracy = {report['accuracy']:.4f}, "
                  f"macro-F1 = {report['macro_f1']:.4f}")
    else:
        for kind in args.models:
            print(f"\n=== {kind.upper()} · subject-wise 5-fold CV ===")
            report = subject_cv_evaluate(X, y, groups, kind=kind)
            reports[kind] = report
            print(f"mean acc = {report['mean_accuracy']:.4f} "
                  f"(±{report['std_accuracy']:.4f}), macro-F1 = {report['macro_f1']:.4f}")

    finish(reports, args)


if __name__ == "__main__":
    main()
