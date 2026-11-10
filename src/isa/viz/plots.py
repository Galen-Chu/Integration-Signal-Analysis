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
        ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Confusion matrix (subject-wise CV)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_subject(per_subject: dict, out_path, title: str = "Per-subject accuracy",
                     model_label: str = ""):
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
    ax.text(mean, len(subjects) - 0.25, f" mean {mean:.1%}",
            color="#52514e", fontsize=9, va="bottom")
    for i, v in enumerate(accs):
        ax.text(v + 0.004, i, f"{v:.0%}", va="center", fontsize=9,
                color="#0b0b0b", zorder=4)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Accuracy")
    ax.xaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")
    ax.set_title(title + (f" · {model_label}" if model_label else ""),
                 fontsize=11)
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
