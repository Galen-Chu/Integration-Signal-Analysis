"""loader_phone 測試——契約驗證＋跨語言 round-trip（node recorder_core → python loader）.

跨語言測試是真正的合約檢驗：fixture 由 app/test/make_fixtures.mjs 用
「錄製頁同一份核心」產生（抖動時間戳、gap 切段、50Hz 重採樣）——python 端
若對契約的理解有任何偏差（欄序、seg 語義、視窗對齊）都會在這裡炸開.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.isa.data.loader_phone import (
    load_phone_csv,
    load_phone_dataset,
    load_phone_session,
    windows_from_rows,
)

REPO = Path(__file__).resolve().parent.parent


def _csv_text(rows):
    lines = ["t_rel_s,seg,bx,by,bz,tx,ty,tz"]
    lines += [",".join(f"{v:.5f}" for v in r) for r in rows]
    return "\n".join(lines) + "\n"


def test_windows_layout_and_count(tmp_path):
    """兩段（各 300 樣本）→ 每段 floor((300-128)/64)+1 = 3 窗；軸布局正確."""
    t = 0.0
    rows = []
    for seg in (1, 2):
        for _ in range(300):
            rows.append([t, seg, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            t += 0.02
        t += 1.0  # seg 間 gap（不檢查間距）
    p = tmp_path / "s.csv"
    p.write_text(_csv_text(rows), encoding="utf-8")
    vals, segs = load_phone_csv(p)
    w = windows_from_rows(vals, segs)
    assert w.shape == (6, 9, 128)  # 3 窗 × 2 段
    assert (w[:, 3:6, :] == 0).all()  # gyro 佔位
    for ax, v in ((0, 1.0), (1, 2.0), (2, 3.0)):  # body xyz
        assert (w[:, ax, :] == v).all()
    for ax, v in ((6, 4.0), (7, 5.0), (8, 6.0)):  # total xyz
        assert (w[:, ax, :] == v).all()


def test_rate_violation_rejected(tmp_path):
    """段內時間步偏離 50Hz → 拒載（錄製頁契約：精確 50Hz 格線）."""
    rows = [[i * 0.03, 1, 0, 0, 0, 0, 0, 0] for i in range(200)]  # 30Hz
    p = tmp_path / "bad.csv"
    p.write_text(_csv_text(rows), encoding="utf-8")
    with pytest.raises(AssertionError, match="50Hz"):
        load_phone_csv(p)


def test_nan_rejected(tmp_path):
    rows = [[i * 0.02, 1, 0, 0, 0, 0, 0, float("nan")] for i in range(200)]
    p = tmp_path / "nan.csv"
    p.write_text(_csv_text(rows), encoding="utf-8")
    with pytest.raises(AssertionError, match="NaN"):
        load_phone_csv(p)


def test_session_and_dataset_load(tmp_path):
    rows = [[i * 0.02, 1, 0.1, 0, 0, 0, 0, 9.81] for i in range(200)]
    (tmp_path / "S01_walk_waist_x.csv").write_text(_csv_text(rows), encoding="utf-8")
    (tmp_path / "S01_walk_waist_x.meta.json").write_text(
        json.dumps(
            {"subject": "S01", "activity": "walk", "position": "waist", "nominal_rate_hz": 50}
        ),
        encoding="utf-8",
    )
    w, label, subject, meta = load_phone_session(tmp_path / "S01_walk_waist_x.csv")
    assert label == "walk" and subject == "S01"
    assert w.shape[1:] == (9, 128) and len(w) == 2  # floor((200-128)/64)+1 = 2

    X, y, g = load_phone_dataset(tmp_path)
    assert X.shape == (2, 9, 128)
    assert set(y) == {"walk"} and set(g) == {"S01"}


def test_meta_rate_mismatch_raises(tmp_path):
    rows = [[i * 0.02, 1, 0, 0, 0, 0, 0, 0] for i in range(200)]
    (tmp_path / "a.csv").write_text(_csv_text(rows), encoding="utf-8")
    (tmp_path / "a.meta.json").write_text(json.dumps({"nominal_rate_hz": 100}), encoding="utf-8")
    with pytest.raises(ValueError, match="rate"):
        load_phone_session(tmp_path / "a.csv")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_cross_language_roundtrip(tmp_path):
    """node recorder_core（真錄製頁核心）產生 fixture → python loader 消費.

    合約斷言：loader 的 stride=128 視窗數 == 錄製頁 windowCount；
    標籤/受試者來自 meta；gap 切段尊重（段內視窗數與 JS 端一致）.
    """
    out = tmp_path / "phone_v2"
    subprocess.run(
        ["node", str(REPO / "app" / "test" / "make_fixtures.mjs"), str(out)],
        check=True,
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    X, y, g = load_phone_dataset(out, stride=128)
    assert X.shape[1:] == (9, 128)
    assert np.isfinite(X).all()
    assert set(y) == {"dance_choreo", "yoga_pose"}
    assert set(g) == {"S99"}

    # 跨語言合約：每個 session 的 loader 視窗數 == JS 端 windowCount（meta 記錄）
    total_expected = 0
    for meta_path in out.glob("*.meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        csv_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".csv"))
        w, _, _, _ = load_phone_session(csv_path, meta_path, stride=128)
        assert len(w) == meta["windows_128"], (
            f"{csv_path.name}: loader {len(w)} ≠ JS {meta['windows_128']}"
        )
        total_expected += meta["windows_128"]
    assert len(X) == total_expected

    # gap 切段被尊重：dance session 的第一段視窗不含第二段的樣本
    vals, segs = load_phone_csv(out / "S99_dance_choreo_chest.csv")
    w_all = windows_from_rows(vals, segs, stride=128)
    first_seg_rows = vals[segs == 1]
    assert np.allclose(w_all[0][0, :], first_seg_rows[:128, 0], atol=1e-5)
