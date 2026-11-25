"""端到端煙霧測試——合成資料走「特徵→模型→評估→視覺化」全管線.

單元測試各自覆蓋模組，這裡補的是**管線層級**的回歸防護：
任何一格的介面改壞（維度、欄序、報告鍵、matplotlib 版本相容）都會在
這裡炸開，而不是等真機跑 scripts/ 才發現。全程合成資料，CI 免下載.
"""

import numpy as np

from src.isa.features.freq_domain import extract_freq_features
from src.isa.features.posture import extract_posture_features
from src.isa.features.time_domain import extract_time_features
from src.isa.features.transient import extract_transient_features
from src.isa.models.train import evaluate_holdout, make_model, subject_cv_evaluate
from src.isa.viz.plots import (
    plot_confusion,
    plot_hr_timeline,
    plot_per_subject,
    plot_position_comparison,
    plot_spectrum,
)


def _synthetic_dataset(n_per_cell=10, seed=0):
    """2 活動 × 4 受試者：(80, 9, 128)＋labels＋groups.

    walking 樣：body 2Hz 振盪、重力朝 z；lying 樣：body 幾乎靜止、重力朝 x
    ——兩活動在時/頻/姿態特徵上都可分，RF 應拿到高分.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(128) / 50
    windows, y, g = [], [], []
    for sid in (101, 102, 103, 104):
        for code, builder in (
            (1, lambda: (0.4 * np.sin(2 * np.pi * 2 * t), [0.0, 0.0, 9.81])),
            (2, lambda: (0.01 * rng.normal(size=128), [9.81, 0.0, 0.0])),
        ):
            for _ in range(n_per_cell):
                body_x, gravity = builder()
                w = np.zeros((9, 128), dtype=np.float32)
                w[0, :] = body_x + 0.02 * rng.normal(size=128)
                w[1, :] = 0.02 * rng.normal(size=128)
                w[2, :] = 0.02 * rng.normal(size=128)
                w[6:9, :] = w[0:3, :] + np.asarray(gravity)[:, None]
                windows.append(w)
                y.append(code)
                g.append(sid)
    return np.stack(windows), np.asarray(y), np.asarray(g)


def _features(windows):
    return np.hstack(
        [
            extract_time_features(windows),
            extract_freq_features(windows),
            extract_posture_features(windows),
            extract_transient_features(windows),
        ]
    )


class TestEndToEnd:
    def test_feature_stack_shapes_and_finite(self):
        windows, y, g = _synthetic_dataset()
        X = _features(windows)
        assert X.shape == (80, 27 + 16 + 6 + 5)
        assert np.isfinite(X).all()

    def test_subject_cv_report_shape(self):
        windows, y, g = _synthetic_dataset()
        X = _features(windows)
        report = subject_cv_evaluate(X, y, g, kind="rf", n_splits=2)
        assert report["n_splits"] == 2
        assert len(report["fold_accuracies"]) == 2
        assert 0.0 <= report["mean_accuracy"] <= 1.0
        cm = np.array(report["confusion_matrix"])
        assert cm.sum() == len(y)
        assert report["n_subjects"] == 4

    def test_synthetic_tasks_are_learnable(self):
        """可分性健全性檢查：兩活動差異明確，CV 準確率應遠高於 0.5.

        若某次重構讓特徵管線悄悄壞掉（欄序錯位、NaN、常數化），
        這條會在真實資料跑之前先炸.
        """
        windows, y, g = _synthetic_dataset()
        X = _features(windows)
        report = subject_cv_evaluate(X, y, g, kind="rf", n_splits=2)
        assert report["mean_accuracy"] > 0.8, report["fold_accuracies"]

    def test_holdout_and_models(self):
        windows, y, g = _synthetic_dataset(n_per_cell=6)  # 4 受試者 × 48 視窗
        X = _features(windows)
        tr, te = np.arange(24), np.arange(24, 48)  # 前 2 受試者訓練、後 2 測試
        assert set(g[tr]).isdisjoint(set(g[te]))
        for kind in ("rf", "svm"):
            r = evaluate_holdout(X[tr], y[tr], X[te], y[te], g[tr], kind=kind)
            assert r["accuracy"] > 0.8
            assert len(r["labels"]) == 2

    def test_make_model_unknown_kind_raises(self):
        import pytest

        with pytest.raises(ValueError, match="unknown model"):
            make_model("bert")


class TestVizSmoke:
    """視覺化層煙霧：matplotlib 版本相容與檔案產出（plots.py 原本無測試）."""

    def _png_ok(self, path):
        assert path.exists() and path.stat().st_size > 1000

    def test_all_plots_render(self, tmp_path):
        cm = np.array([[8, 2], [1, 9]])
        plot_confusion(cm, [1, 2], ["WALKING", "SITTING"], tmp_path / "confusion.png")
        self._png_ok(tmp_path / "confusion.png")

        plot_per_subject({101: 0.9, 102: 0.7, 103: 0.85}, tmp_path / "subj.png")
        self._png_ok(tmp_path / "subj.png")

        plot_spectrum(np.sin(2 * np.pi * 2 * np.arange(128) / 50), 50, tmp_path / "spec.png")
        self._png_ok(tmp_path / "spec.png")

        plot_hr_timeline(np.array([70.0, 90.0, 120.0]), np.arange(3.0), tmp_path / "hr.png")
        self._png_ok(tmp_path / "hr.png")

    def test_position_comparison_plot(self, tmp_path):
        """位置對照圖：以迷你報告（3 類混淆）走完整繪圖路徑."""

        def report(acc):
            return {
                "mean_accuracy": acc,
                "std_accuracy": 0.03,
                "confusion_matrix": [[9, 1, 0], [2, 7, 1], [0, 2, 8]],
                "labels": [1, 2, 3],
            }

        results = {
            pos: {"rf": report(a), "svm": report(a + 0.01)}
            for pos, a in (("hand", 0.75), ("chest", 0.74), ("ankle", 0.69))
        }
        out = tmp_path / "position.png"
        plot_position_comparison(
            results,
            out,
            {1: "lying", 2: "sitting", 3: "standing"},
            static=("lying", "sitting", "standing"),
        )
        self._png_ok(out)
