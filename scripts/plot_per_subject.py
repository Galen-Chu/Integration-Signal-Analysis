"""per-subject 準確率分布圖 · 受試者異質性視覺化（N4）.

    python scripts/plot_per_subject.py                # ours2 × SVM（主軸）
    python scripts/plot_per_subject.py --features ours2 --models svm

官方 split（21 人訓練→9 人測試）：折間變異 86-98% 的文字敘述，
在這裡變成「哪些測試受試者難」的證據——GroupKFold 鐵律的展示素材.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.isa.config import FIGURES_DIR, UCI_DIR  # noqa: E402
from src.isa.data.loader import load_labels, load_raw_signals, load_subjects  # noqa: E402
from src.isa.features.freq_domain import extract_freq_features  # noqa: E402
from src.isa.features.posture import extract_posture_features  # noqa: E402
from src.isa.features.time_domain import extract_time_features  # noqa: E402
from src.isa.models.train import make_model  # noqa: E402
from src.isa.viz.plots import plot_per_subject  # noqa: E402


def build_ours2(split: str) -> np.ndarray:
    windows = load_raw_signals(UCI_DIR, split)
    return np.hstack([extract_time_features(windows),
                      extract_freq_features(windows),
                      extract_posture_features(windows)])


def main():
    parser = argparse.ArgumentParser(description="Per-subject accuracy 圖")
    parser.add_argument("--features", choices=["ours2"], default="ours2")
    parser.add_argument("--models", nargs="+", default=["svm"])
    args = parser.parse_args()

    X = build_ours2("train")
    y = load_labels(UCI_DIR, "train")
    X_te = build_ours2("test")
    y_te = load_labels(UCI_DIR, "test")
    subjects_te = load_subjects(UCI_DIR, "test")
    print(f"official holdout: train {X.shape} → test {X_te.shape}, "
          f"{len(np.unique(subjects_te))} test subjects")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for kind in args.models:
        model = make_model(kind)
        model.fit(X, y)
        pred = model.predict(X_te)
        per_subject = {}
        for s in np.unique(subjects_te):
            mask = subjects_te == s
            per_subject[int(s)] = float((pred[mask] == y_te[mask]).mean())
        for s in sorted(per_subject):
            print(f"  subject {s:2d}: {per_subject[s]:.4f} "
                  f"({int((subjects_te == s).sum())} windows)")
        out = FIGURES_DIR / f"per_subject_{kind}_official_{args.features}.png"
        plot_per_subject(per_subject, out,
                         title="Per-subject accuracy (official test split)",
                         model_label=f"{args.features} · {kind.upper()}")
        print(f"figure → {out}")


if __name__ == "__main__":
    main()
