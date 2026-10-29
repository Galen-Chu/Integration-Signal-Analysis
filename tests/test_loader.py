"""UCI loader 測試——tmp_path 迷你 fixture，免下載真實資料."""
import numpy as np
import pytest

from src.isa.data.loader import (SIGNALS, load_labels, load_raw_signals,
                                 load_subjects)


@pytest.fixture
def mini_uci(tmp_path):
    """造一個 3 視窗×3 受試者的迷你 UCI 目錄結構."""
    root = tmp_path / "UCI HAR Dataset"
    for split, n in (("train", 3), ("test", 2)):
        signals_dir = root / split / "Inertial Signals"
        signals_dir.mkdir(parents=True)
        for sig in SIGNALS:
            data = np.arange(n * 128, dtype=float).reshape(n, 128) / 1000
            np.savetxt(signals_dir / f"{sig}_{split}.txt", data, fmt="%.6f")
        labels = np.arange(1, n + 1).reshape(n, 1)      # 1..n，皆為合法活動碼
        np.savetxt(root / split / f"y_{split}.txt", labels, fmt="%d")
        subjects = (np.arange(n) + 1).reshape(n, 1)     # 受試者 1..n
        np.savetxt(root / split / f"subject_{split}.txt", subjects, fmt="%d")
    return root


def test_raw_signals_shape_and_dtype(mini_uci):
    X = load_raw_signals(mini_uci, "train")
    assert X.shape == (3, 9, 128)
    assert X.dtype == np.float32


def test_raw_signals_values_passthrough(mini_uci):
    X = load_raw_signals(mini_uci, "train")
    assert X[0, 0, 0] == pytest.approx(0.0)
    assert X[0, 0, -1] == pytest.approx(127 / 1000)


def test_labels(mini_uci):
    y = load_labels(mini_uci, "train")
    assert y.tolist() == [1, 2, 3]
    assert y.dtype == int


def test_subjects(mini_uci):
    s = load_subjects(mini_uci, "train")
    assert s.tolist() == [1, 2, 3]


def test_labels_reject_unknown_code(mini_uci):
    np.savetxt(mini_uci / "train" / "y_train.txt",
               np.array([[7]]), fmt="%d")
    with pytest.raises(AssertionError):
        load_labels(mini_uci, "train")
