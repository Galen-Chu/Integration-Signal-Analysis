"""run_pamap2_position · IMU 位置對照實驗——感測器位置假說的受控驗證.

    python scripts/run_pamap2_position.py
問題：findings 的「感測器位置比特徵工程更影響靜態動作識別」過去是從手部
IMU 的結果**推論**的，驗證要等 v2 自採資料。但 PAMAP2 的每位受試者同時
配戴三個 IMU（手/胸/踝）——用同一套 49 維特徵與 GroupKFold 協議對照三
位置，零採集成本即可把推論升級為受控對照實驗。

只動「位置」一個變數：同受試者、同活動、同特徵、同模型、同協議。
加速度計顆粒用 ±6g——±16g 顆粒的 chest x 軸與 ankle y 軸為已知故障
通道（9/9 受試者凍結），±6g 三位置全軸健康（詳見 loader 註記）。
手部 ±16g 既有基線（report_pamap2_ours2.json）保留作為連續性錨點.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.isa.config import FIGURES_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402
from src.isa.data.loader_pamap2 import PAMAP2_ACTIVITIES, load_pamap2_windows  # noqa: E402
from src.isa.features.freq_domain import extract_freq_features  # noqa: E402
from src.isa.features.posture import extract_posture_features  # noqa: E402
from src.isa.features.time_domain import extract_time_features  # noqa: E402
from src.isa.models.train import subject_cv_evaluate  # noqa: E402
from src.isa.viz.plots import plot_position_comparison  # noqa: E402

STATIC = ("lying", "sitting", "standing")
SCALE = "6g"


def static_recalls(report: dict) -> dict:
    """由摺疊加總的混淆矩陣算靜態活動召回（分類複現的檢驗核心）."""
    labels = report["labels"]
    code_of = {v: k for k, v in PAMAP2_ACTIVITIES.items()}
    cm = np.array(report["confusion_matrix"])
    recall = dict(zip(labels, cm.diagonal() / cm.sum(axis=1), strict=True))
    return {name: float(recall[code_of[name]]) for name in STATIC}


def run_position(data_dir, position: str) -> dict:
    windows, y, groups = load_pamap2_windows(data_dir, position=position,
                                             scale=SCALE)
    counts = {PAMAP2_ACTIVITIES[k]: int(v)
              for k, v in zip(*np.unique(y, return_counts=True), strict=True)}
    print(f"\n=== {position.upper()} (±{SCALE} acc) === windows: {windows.shape}")
    print("windows/activity:", counts)

    X = np.hstack([extract_time_features(windows),
                   extract_freq_features(windows),
                   extract_posture_features(windows)])
    print(f"features: {X.shape[1]}-dim (ours2), finite: {bool(np.isfinite(X).all())}")

    reports = {}
    for kind in ("rf", "svm"):
        print(f"\n--- {kind.upper()} · subject-wise 5-fold CV ---")
        report = subject_cv_evaluate(X, y, groups, kind=kind)
        reports[kind] = report
        print(f"mean acc = {report['mean_accuracy']:.4f} "
              f"(±{report['std_accuracy']:.4f}), macro-F1 = {report['macro_f1']:.4f}")

    reports["static_recall_svm"] = static_recalls(reports["svm"])
    print(f"\n靜態召回（SVM）："
          f"{ {k: round(v, 3) for k, v in reports['static_recall_svm'].items()} }")
    return reports


def main():
    data_dir = RAW_DIR / "PAMAP2_Dataset" / "Protocol"
    if not data_dir.is_dir():
        raise SystemExit("PAMAP2 未下載——見 README（download_pamap2()）")

    results = {pos: run_position(data_dir, pos)
               for pos in ("hand", "chest", "ankle")}

    print("\n========== 位置對照摘要（subject-wise CV） ==========")
    print(f"{'':8s} {'RF acc':>15s} {'SVM acc':>15s} "
          f"{'lying':>7s} {'sitting':>7s} {'standing':>8s}  (SVM 靜態召回)")
    for pos, rep in results.items():
        rf, svm = rep["rf"], rep["svm"]
        sr = rep["static_recall_svm"]
        print(f"{pos:8s} {rf['mean_accuracy']:>7.1%}±{rf['std_accuracy']:.1%} "
              f"{svm['mean_accuracy']:>7.1%}±{svm['std_accuracy']:.1%} "
              f"{sr['lying']:>7.1%} {sr['sitting']:>7.1%} {sr['standing']:>8.1%}")

    fig_path = FIGURES_DIR / "pamap2_position_comparison.png"
    plot_position_comparison(results, fig_path, PAMAP2_ACTIVITIES)
    print(f"\nfigure → {fig_path}")

    out = PROCESSED_DIR / "report_pamap2_position.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"report → {out}")


if __name__ == "__main__":
    main()
