"""PAMAP2 loader 測試——迷你合成 .dat."""
import numpy as np
import pytest

from src.isa.data.loader_pamap2 import load_all_protocol, load_subject


def _write_dat(path, rows):
    """每列補齊 54 欄（loader 只取前 7 欄）."""
    full = np.zeros((len(rows), 54))
    for i, r in enumerate(rows):
        full[i, :7] = r
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        np.savetxt(fh, full, fmt="%.4f")


@pytest.fixture
def mini_dir(tmp_path):
    # subject101：lying(1) t=1-3s hr=70 → running(5) t=4-6s hr=150
    rows = (
        [[1.0, 1, 70.0, 30.0, 0.0, 0.0, 1.0],
         [2.0, 1, 70.0, 30.0, 0.0, 0.0, 1.0],
         [3.0, 1, 70.0, 30.0, 0.0, 0.0, 1.0]]
        + [[4.0, 5, 150.0, 31.0, 1.0, 0.0, 0.0],
           [5.0, 5, 150.0, 31.0, 1.0, 0.0, 0.0],
           [6.0, 5, 150.0, 31.0, 1.0, 0.0, 0.0]]
    )
    _write_dat(tmp_path / "subject101.dat", rows)
    return tmp_path


def test_load_subject_resamples_and_drops_transient(mini_dir):
    df = load_subject(mini_dir / "subject101.dat")
    assert len(df) == 6            # 6 秒（transient 已排除）
    assert set(df["activity"]) == {1, 5}
    assert df["hr"].tolist() == [70.0] * 3 + [150.0] * 3


def test_load_subject_ffills_hr_gaps(tmp_path):
    rows = [
        [1.0, 4, 120.0, 30.0, 0, 0, 1],
        [2.0, 4, np.nan, 30.0, 0, 0, 1],   # HR 缺值 → 前向填充
    ]
    _write_dat(tmp_path / "subject101.dat", rows)
    df = load_subject(tmp_path / "subject101.dat")
    assert df["hr"].tolist() == [120.0, 120.0]


def test_acc_magnitude(tmp_path):
    rows = [[1.0, 2, 80.0, 30.0, 3.0, 4.0, 0.0]]   # mag = 5
    _write_dat(tmp_path / "subject101.dat", rows)
    df = load_subject(tmp_path / "subject101.dat")
    assert df.loc[0, "acc_mag"] == pytest.approx(5.0)


def test_load_all_protocol_adds_subject(mini_dir):
    df = load_all_protocol(mini_dir)
    assert "subject" in df.columns
    assert set(df["subject"]) == {101}
