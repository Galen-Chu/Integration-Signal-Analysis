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


class TestWindows:
    def _write_block_dat(self, tmp_path, sid=101):
        """造 4 秒×2 區塊（lying 1、sitting 2）@100Hz：DC 重力＋小正弦."""
        import numpy as np
        n = 400
        t = np.arange(n) / 100.0
        acc = np.zeros((2 * n, 3))
        acc[:n] = np.column_stack([0.05 * np.sin(2 * np.pi * t[:n]),
                                   0.05 * np.sin(2 * np.pi * t[:n]),
                                   np.full(n, 9.81)])
        acc[n:] = np.column_stack([0.05 * np.sin(2 * np.pi * t),
                                   0.05 * np.sin(2 * np.pi * t),
                                   np.full(n, 9.0)])         # 坐姿：重力偏一點
        acts = np.r_[np.full(n, 1), np.full(n, 2)]
        rows = np.zeros((2 * n, 54))
        rows[:, 1] = acts
        rows[:, 4:7] = acc
        path = tmp_path / f"subject{sid}.dat"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            np.savetxt(fh, rows, fmt="%.4f")
        return tmp_path

    def test_windows_labels_subjects(self, tmp_path):
        from src.isa.data.loader_pamap2 import load_pamap2_windows
        self._write_block_dat(tmp_path)
        w, y, g = load_pamap2_windows(tmp_path)
        assert w.shape[1:] == (9, 128)
        assert set(np.unique(y)) == {1, 2}          # lying＋sitting
        assert set(np.unique(g)) == {101}
        # 每區塊 400 列 → 200 樣本@50Hz → 步長 64：floor((200-128)/64)+1 = 2 窗/區塊
        assert len(y) == 4

    def test_gravity_removed_from_body(self, tmp_path):
        from src.isa.data.loader_pamap2 import load_pamap2_windows
        self._write_block_dat(tmp_path)
        w, y, g = load_pamap2_windows(tmp_path)
        # body（軸 0-2）應已除 DC：均值≈0；total（軸 6-8）保留重力 ≈9.81
        assert abs(w[0][2, :].mean()) < 0.5         # body_z 均值≈0
        assert 9.5 < w[0][8, :].mean() < 10.1       # total_z ≈ 9.81
