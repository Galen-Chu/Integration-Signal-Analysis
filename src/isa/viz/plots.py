"""Visualization · 混淆矩陣、頻譜、心率時間軸."""

import matplotlib

matplotlib.use("Agg")  # 無顯示環境也能出圖
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion(cm: np.ndarray, labels: list, names: list, out_path):
    """names: 與 labels 對應的活動名稱（順序一致）."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(
        ax=ax, cmap="Blues", colorbar=True, values_format="d"
    )
    ax.set_title("Confusion matrix (subject-wise CV)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_subject(
    per_subject: dict, out_path, title: str = "Per-subject accuracy", model_label: str = ""
):
    """per_subject: {subject_id: accuracy}——受試者異質性視覺化.

    單一系列（各受試者準確率）→ 排序長條＋均值線＋直接標值；
    軸與格線退後，讓「哪些人難」一眼可見（dataviz 規格）.
    """
    items = sorted(per_subject.items(), key=lambda kv: kv[1])
    subjects = [f"S{s:02d}" for s, _ in items]
    accs = [v for _, v in items]
    mean = float(np.mean(accs))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(subjects, accs, height=0.62, color="#167839", zorder=3)
    ax.axvline(mean, color="#52514e", lw=1.2, ls="--", zorder=4)
    ax.text(
        mean, len(subjects) - 0.25, f" mean {mean:.1%}", color="#52514e", fontsize=9, va="bottom"
    )
    for i, v in enumerate(accs):
        ax.text(v + 0.004, i, f"{v:.0%}", va="center", fontsize=9, color="#0b0b0b", zorder=4)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Accuracy")
    ax.xaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")
    ax.set_title(title + (f" · {model_label}" if model_label else ""), fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, lw=0.5, color="#e1e0d9", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spectrum(x: np.ndarray, fs: int, out_path, title: str = "Spectrum"):
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(freqs, spectrum, lw=1)
    ax.set_xlabel("Hz")
    ax.set_ylabel("|FFT|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hr_timeline(hr: np.ndarray, times, out_path, title: str = "Heart rate"):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(times, hr, lw=0.8)
    ax.set_ylabel("bpm")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# dataviz 參考調色盤類別槽 1-3（前三槽通過 all-pairs CVD 驗證）＋墨色/格線 token
POSITION_COLORS = {"hand": "#2a78d6", "chest": "#eb6834", "ankle": "#1baf7a"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"


def plot_position_comparison(
    results: dict, out_path, activity_names: dict, static=("lying", "sitting", "standing")
):
    """IMU 位置對照圖：左＝總體準確率（×模型）、右＝靜態召回（SVM）.

    results: {position: {"rf": report, "svm": report}}——subject_cv_evaluate 輸出；
    activity_names: 活動代碼→名稱（呼叫端提供，viz 不依賴 data 層）.
    全圖共用「位置」一種顏色語義（dataviz：顏色跟著實體），單一圖例.
    """
    positions = [p for p in ("hand", "chest", "ankle") if p in results]
    models = [m for m in ("rf", "svm") if m in results[positions[0]]]
    name_of = {"hand": "Hand", "chest": "Chest", "ankle": "Ankle"}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2))

    # 左：總體準確率——x=模型、每組三條（位置）；誤差線＝折間 std；
    # 數值標籤放誤差線上緣之上，避免打架
    x = np.arange(len(models))
    w = 0.24
    for i, pos in enumerate(positions):
        means = [results[pos][m]["mean_accuracy"] for m in models]
        stds = [results[pos][m]["std_accuracy"] for m in models]
        bars = ax_a.bar(
            x + (i - 1) * w,
            means,
            width=w,
            color=POSITION_COLORS[pos],
            label=name_of[pos],
            zorder=3,
        )
        ax_a.errorbar(
            x + (i - 1) * w,
            means,
            yerr=stds,
            fmt="none",
            ecolor=INK_2,
            elinewidth=1,
            capsize=2,
            zorder=4,
        )
        for rect, mean, std in zip(bars, means, stds, strict=True):
            ax_a.text(
                rect.get_x() + rect.get_width() / 2,
                mean + std + 0.032,
                f"{mean:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
    ax_a.set_xticks(x, [m.upper() for m in models])
    ax_a.set_ylim(0, 1.0)
    ax_a.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    ax_a.set_title("Overall accuracy (12 activities)", fontsize=10.5)

    # 右：靜態召回（SVM 總和混淆矩陣）——x=活動、每組三條（位置）
    svm_labels = results[positions[0]]["svm"]["labels"]
    code_of = {v: k for k, v in activity_names.items()}
    x2 = np.arange(len(static))
    for i, pos in enumerate(positions):
        cm = np.array(results[pos]["svm"]["confusion_matrix"])
        recall = dict(zip(svm_labels, cm.diagonal() / cm.sum(axis=1), strict=True))
        vals = [recall[code_of[a]] for a in static]
        bars = ax_b.bar(
            x2 + (i - 1) * w,
            vals,
            width=w,
            color=POSITION_COLORS[pos],
            label=name_of[pos],
            zorder=3,
        )
        ax_b.bar_label(bars, labels=[f"{v:.0%}" for v in vals], fontsize=8, color=INK, padding=2)
    ax_b.set_xticks(x2, [a.replace("_", " ") for a in static])
    ax_b.set_ylim(0, 1.12)
    ax_b.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    ax_b.set_title("Static recall · SVM (the confusion that matters)", fontsize=10.5)

    for ax in (ax_a, ax_b):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.yaxis.grid(True, lw=0.5, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, labelsize=9)

    fig.suptitle(
        "PAMAP2 · IMU position comparison (subject-wise 5-fold CV)", fontsize=11.5, color=INK, y=1.0
    )
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
