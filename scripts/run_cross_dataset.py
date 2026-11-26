"""run_cross_dataset · 跨資料集遷移實驗——findings 限制 #2 的實證.

    python scripts/run_cross_dataset.py
問題：UCI HAR 訓練的分類模型搬到 PAMAP2 還剩多少？兩資料集六個活動
一一對應（walk/上樓/下樓/坐/站/躺），同 49 維特徵，只動「資料來源」。

對照設計（分離 domain shift 的成分）：
  transfer     UCI 21 人訓練 → PAMAP2 六類全部評估（跨域）
  in-dist 上界 PAMAP2 六類 subject-wise CV（同域、同任務）
  UCI 錨點     UCI subject-wise CV（同域基準，文獻數字 89.2%）

感測器與位置差異（腰側手機 vs 腕戴 ±16g）是 domain shift 的一部分——
這是實驗要量測的對象，不是要消除的混淆.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.isa.config import PROCESSED_DIR, RAW_DIR, UCI_DIR  # noqa: E402
from src.isa.data.loader import load_labels, load_raw_signals, load_subjects  # noqa: E402
from src.isa.data.loader_pamap2 import load_pamap2_windows  # noqa: E402
from src.isa.features.freq_domain import extract_freq_features  # noqa: E402
from src.isa.features.posture import extract_posture_features  # noqa: E402
from src.isa.features.time_domain import extract_time_features  # noqa: E402
from src.isa.models.train import evaluate_holdout, subject_cv_evaluate  # noqa: E402

# UCI 活動代碼 → PAMAP2 活動代碼（兩資料集的六類對應）
UCI_TO_PAMAP2 = {1: 4, 2: 12, 3: 13, 4: 2, 5: 3, 6: 1}
UCI_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def ours2_features(windows: np.ndarray) -> np.ndarray:
    return np.hstack(
        [
            extract_time_features(windows),
            extract_freq_features(windows),
            extract_posture_features(windows),
        ]
    )


def per_class_recall(cm: np.ndarray, labels: list) -> dict:
    return {
        int(c): float(cm[i][i] / cm[i].sum()) if cm[i].sum() else 0.0 for i, c in enumerate(labels)
    }


def main():
    print("Loading UCI train (21 subjects)…")
    win_uci = load_raw_signals(UCI_DIR, "train")
    y_uci = load_labels(UCI_DIR, "train")
    g_uci = load_subjects(UCI_DIR, "train")
    X_uci = ours2_features(win_uci)
    print(f"UCI: {X_uci.shape}")

    print("Loading PAMAP2 windows (hand ±16g baseline)…")
    win_p, y_p, g_p = load_pamap2_windows(RAW_DIR / "PAMAP2_Dataset" / "Protocol")
    keep = np.isin(y_p, list(UCI_TO_PAMAP2.values()))
    win_p, y_p, g_p = win_p[keep], y_p[keep], g_p[keep]
    # PAMAP2 代碼 → UCI 代碼（共同標籤空間）
    remap = {v: k for k, v in UCI_TO_PAMAP2.items()}
    y_p_common = np.asarray([remap[c] for c in y_p])
    print(f"PAMAP2 mapped: {win_p.shape[0]} windows, {len(np.unique(g_p))} subjects")
    X_p = ours2_features(win_p)
    print(f"features: {X_p.shape[1]}-dim")

    results = {
        "mapping": {str(k): UCI_NAMES[k] for k in UCI_TO_PAMAP2},
        "uci_train_windows": int(len(y_uci)),
        "pamap2_mapped_windows": int(len(y_p_common)),
    }

    print("\n=== transfer：UCI 訓練 → PAMAP2 六類評估 ===")
    results["transfer"] = {}
    for kind in ("rf", "svm"):
        r = evaluate_holdout(X_uci, y_uci, X_p, y_p_common, g_uci, kind=kind)
        recalls = per_class_recall(np.array(r["confusion_matrix"]), r["labels"])
        results["transfer"][kind] = {
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "recalls": recalls,
        }
        print(f"{kind.upper()}: acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f}")
        for c in sorted(recalls):
            print(f"    {UCI_NAMES[c]:<18} recall={recalls[c]:.3f}")

    print("\n=== in-dist 上界：PAMAP2 六類 subject-wise CV ===")
    results["pamap2_in_dist"] = {}
    for kind in ("rf", "svm"):
        rep = subject_cv_evaluate(X_p, y_p_common, g_p, kind=kind)
        recalls = per_class_recall(np.array(rep["confusion_matrix"]), rep["labels"])
        results["pamap2_in_dist"][kind] = {
            "mean_accuracy": rep["mean_accuracy"],
            "std_accuracy": rep["std_accuracy"],
            "macro_f1": rep["macro_f1"],
            "recalls": recalls,
        }
        print(
            f"{kind.upper()}: acc={rep['mean_accuracy']:.4f}"
            f"±{rep['std_accuracy']:.4f} f1={rep['macro_f1']:.4f}"
        )

    print("\n=== UCI 錨點：subject-wise CV（同域基準） ===")
    results["uci_anchor"] = {}
    for kind in ("rf", "svm"):
        rep = subject_cv_evaluate(X_uci, y_uci, g_uci, kind=kind)
        results["uci_anchor"][kind] = {
            "mean_accuracy": rep["mean_accuracy"],
            "std_accuracy": rep["std_accuracy"],
            "macro_f1": rep["macro_f1"],
        }
        print(f"{kind.upper()}: acc={rep['mean_accuracy']:.4f}±{rep['std_accuracy']:.4f}")

    # 摘要
    print("\n========== 遷移落差摘要（SVM） ==========")
    t = results["transfer"]["svm"]["accuracy"]
    i = results["pamap2_in_dist"]["svm"]["mean_accuracy"]
    u = results["uci_anchor"]["svm"]["mean_accuracy"]
    print(f"UCI 同域 {u:.1%} → PAMAP2 同域 {i:.1%} → 跨域 {t:.1%}")
    print(f"domain shift 成分（PAMAP2 同任務內外差）＝ {i - t:+.1%}")
    results["summary_svm"] = {
        "uci_in_dist": u,
        "pamap2_in_dist": i,
        "transfer": t,
        "domain_shift_gap": i - t,
    }

    out = PROCESSED_DIR / "cross_dataset_transfer.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
