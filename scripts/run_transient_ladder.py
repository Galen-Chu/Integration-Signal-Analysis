"""Transient ladder · 姿態轉換暫態特徵的增量驗證（N2 之後的第二波）.

    python scripts/run_transient_ladder.py

假說（findings 第一節指出的方向）：SITTING↔STANDING 殘餘混淆來自「視窗
內不變的靜態訊號」，改善方向是視窗**內的變化**（姿態轉換暫態、慢擺）——
+5 維暫態特徵（src/isa/features/transient.py）。

成功條件（預先聲明，避免事後挑數字）：
  主準則  官方 split SVM SITTING 召回 79.6% → ≥85%（眺望 89%＝官方 561 維水準）
          且 overall 90.4% 不退步逾 0.5pp
  副準則  GroupKFold CV mean acc 89.2% → 不降逾 1pp、SITTING 召回方向一致
  未達主準則 → 負結果記錄，49 維維持精簡集最適（同 N2 處理）
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
from src.isa.features.freq_domain import extract_freq_features  # noqa: E402
from src.isa.features.posture import extract_posture_features  # noqa: E402
from src.isa.features.time_domain import extract_time_features  # noqa: E402
from src.isa.features.transient import extract_transient_features  # noqa: E402
from src.isa.models.train import evaluate_holdout, subject_cv_evaluate  # noqa: E402

SIT = 4


def per_class_recall(cm: list, labels: list, code: int) -> float:
    i = labels.index(code)
    return float(cm[i][i] / sum(cm[i])) if sum(cm[i]) else 0.0


def main():
    win_tr = load_raw_signals(UCI_DIR, "train")
    win_te = load_raw_signals(UCI_DIR, "test")
    y, y_te = load_labels(UCI_DIR, "train"), load_labels(UCI_DIR, "test")
    groups = load_subjects(UCI_DIR, "train")

    def ours2(windows):
        return np.hstack(
            [
                extract_time_features(windows),
                extract_freq_features(windows),
                extract_posture_features(windows),
            ]
        )

    base_tr, base_te = ours2(win_tr), ours2(win_te)
    trans_tr, trans_te = (extract_transient_features(win_tr), extract_transient_features(win_te))
    up_tr, up_te = (np.hstack([base_tr, trans_tr]), np.hstack([base_te, trans_te]))
    uci_tr, uci_te = (load_uci_features(UCI_DIR, "train"), load_uci_features(UCI_DIR, "test"))
    print(
        f"ours2: {base_tr.shape[1]}d → +transient: {up_tr.shape[1]}d "
        f"(uci anchor: {uci_tr.shape[1]}d)"
    )

    results = {}
    print("\n=== 官方 split（21 訓練 → 9 測試） ===")
    print(f"{'step':<22} {'dims':>4} | model | acc      SIT-recall  macro-F1")
    for name, X, X_te in [
        ("ours2 (baseline)", base_tr, base_te),
        ("ours2+transient", up_tr, up_te),
        ("uci-561 (anchor)", uci_tr, uci_te),
    ]:
        results[name] = {"dims": int(X.shape[1])}
        for m in ("svm", "rf", "xgb"):
            r = evaluate_holdout(X, y, X_te, y_te, groups, kind=m)
            sit = per_class_recall(r["confusion_matrix"], r["labels"], SIT)
            results[name][m] = {
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "sit_recall": sit,
            }
            print(
                f"{name:<22} {X.shape[1]:>4} | {m.upper():<5} | "
                f"{r['accuracy']:.4f}  {sit:.3f}      {r['macro_f1']:.4f}"
            )

    print("\n=== GroupKFold 5 折 CV（受試者零重疊） ===")
    for name, X in [("ours2 (baseline)", base_tr), ("ours2+transient", up_tr)]:
        results[name]["cv"] = {}
        for m in ("svm", "rf"):
            rep = subject_cv_evaluate(X, y, groups, kind=m)
            sit = per_class_recall(rep["confusion_matrix"], rep["labels"], SIT)
            results[name]["cv"][m] = {
                "mean_accuracy": rep["mean_accuracy"],
                "std_accuracy": rep["std_accuracy"],
                "sit_recall": sit,
                "macro_f1": rep["macro_f1"],
            }
            print(
                f"{name:<22} | {m.upper():<3} | {rep['mean_accuracy']:.4f} "
                f"±{rep['std_accuracy']:.4f}  SIT-recall {sit:.3f}"
            )

    # 預先聲明的判讀
    a = results["ours2 (baseline)"]["svm"]
    b = results["ours2+transient"]["svm"]
    d_sit = b["sit_recall"] - a["sit_recall"]
    d_acc = b["accuracy"] - a["accuracy"]
    verdict = (
        "POSITIVE——暫態特徵有效"
        if (b["sit_recall"] >= 0.85 and d_acc >= -0.005)
        else "MIXED——SITTING 有動但不達門檻"
        if d_sit >= 0.02 and d_acc >= -0.005
        else "NEGATIVE——同 N2：49 維維持精簡集最適"
    )
    print(f"\n判讀：ΔSIT={d_sit:+.3f}, Δacc={d_acc:+.4f} → {verdict}")
    results["verdict"] = {
        "delta_sit_recall": d_sit,
        "delta_accuracy": d_acc,
        "rule": "主準則 SIT≥85% 且 acc 不退 >0.5pp",
        "verdict": verdict,
    }

    out = PROCESSED_DIR / "transient_ladder.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report → {out}")


if __name__ == "__main__":
    main()
