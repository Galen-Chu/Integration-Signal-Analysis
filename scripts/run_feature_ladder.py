"""Feature ladder · N2 選擇性特徵移植的增量驗證.

    python scripts/run_feature_ladder.py

假說：ours2(49 維) 的剩餘差距（官方 split SVM 90.4% vs 95.5%）集中在
分佈形狀資訊。逐家族補上 ECDF 百分位、頻帶能量，每階官方 split 重評——
每族買到幾 pp、維度代價多少，數字說話；無貢獻的家族即捨棄.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.isa.config import PROCESSED_DIR, UCI_DIR  # noqa: E402
from src.isa.data.loader import (  # noqa: E402
    load_labels,
    load_raw_signals,
    load_subjects,
    load_uci_features,
)
from src.isa.features.distribution import extract_band_features, extract_ecdf_features  # noqa: E402
from src.isa.features.freq_domain import extract_freq_features  # noqa: E402
from src.isa.features.posture import extract_posture_features  # noqa: E402
from src.isa.features.time_domain import extract_time_features  # noqa: E402
from src.isa.models.train import evaluate_holdout  # noqa: E402

SIT, STAND = 4, 5  # 活動代碼：靜態姿勢混淆是我們的靶


def per_class_recall(cm: list, labels: list, code: int) -> float:
    i = labels.index(code)
    return float(cm[i][i] / sum(cm[i])) if sum(cm[i]) else 0.0


def main():
    win_tr = load_raw_signals(UCI_DIR, "train")
    win_te = load_raw_signals(UCI_DIR, "test")
    y, y_te = load_labels(UCI_DIR, "train"), load_labels(UCI_DIR, "test")
    groups = load_subjects(UCI_DIR, "train")

    base_tr = np.hstack([extract_time_features(win_tr), extract_freq_features(win_tr),
                         extract_posture_features(win_tr)])
    base_te = np.hstack([extract_time_features(win_te), extract_freq_features(win_te),
                         extract_posture_features(win_te)])
    ecdf_tr, ecdf_te = extract_ecdf_features(win_tr), extract_ecdf_features(win_te)
    band_tr, band_te = extract_band_features(win_tr), extract_band_features(win_te)
    uci_tr, uci_te = load_uci_features(UCI_DIR, "train"), load_uci_features(UCI_DIR, "test")

    # jerk 假說：官方 561 維含整個 jerk（加速度微分）統計家族，我們完全沒用。
    # 現有時域/頻域萃取器只吃 windows[:,0:3,:]——把差分訊號放進 0-2 軸即重用。
    def jerk_feats(windows):
        jerk = np.zeros_like(windows)
        jerk[:, 0:3, :] = np.diff(windows[:, 0:3, :], axis=-1, prepend=0.0)
        return np.hstack([extract_time_features(jerk), extract_freq_features(jerk)])

    jerk_tr, jerk_te = jerk_feats(win_tr), jerk_feats(win_te)
    print(f"base(ours2): {base_tr.shape[1]}d  +ecdf: {ecdf_tr.shape[1]}d  "
          f"+bands: {band_tr.shape[1]}d  +jerk: {jerk_tr.shape[1]}d  "
          f"(uci anchor: {uci_tr.shape[1]}d)")

    steps = [
        ("ours2 (baseline)", base_tr, base_te),
        ("ours2+ecdf", np.hstack([base_tr, ecdf_tr]), np.hstack([base_te, ecdf_te])),
        ("ours2+bands", np.hstack([base_tr, band_tr]), np.hstack([base_te, band_te])),
        ("ours2+ecdf+bands", np.hstack([base_tr, ecdf_tr, band_tr]),
         np.hstack([base_te, ecdf_te, band_te])),
        ("ours2+jerk", np.hstack([base_tr, jerk_tr]), np.hstack([base_te, jerk_te])),
        ("ours2+jerk+bands", np.hstack([base_tr, jerk_tr, band_tr]),
         np.hstack([base_te, jerk_te, band_te])),
        ("uci-561 (anchor)", uci_tr, uci_te),
    ]
    models = ["svm", "rf", "xgb"]
    results = {}
    print(f"\n{'step':<20} {'dims':>5} | " + " | ".join(
        f"{m.upper()} acc / SIT-recall" for m in models))
    print("-" * 78)
    for name, X, X_te in steps:
        results[name] = {"dims": int(X.shape[1])}
        cells = []
        for m in models:
            r = evaluate_holdout(X, y, X_te, y_te, groups, kind=m)
            sit = per_class_recall(r["confusion_matrix"], r["labels"], SIT)
            results[name][m] = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                                "sit_recall": sit}
            cells.append(f"{r['accuracy']:.4f} / {sit:.3f}")
        print(f"{name:<20} {X.shape[1]:>5} | " + " | ".join(cells))

    out = PROCESSED_DIR / "feature_ladder_official.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
