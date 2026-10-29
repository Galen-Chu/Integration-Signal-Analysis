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
