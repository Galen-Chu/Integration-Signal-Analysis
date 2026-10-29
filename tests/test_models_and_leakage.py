"""洩漏防護與模型測試——subject-wise 切分是不可動搖的根基."""
import numpy as np
from sklearn.model_selection import GroupKFold

from src.isa.models.train import make_model, subject_cv_evaluate


def _synthetic_dataset(n_subjects=20, per_subject=10, seed=0):
    """可分類的合成資料：class 由受試者無關的訊號規律決定."""
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for s in range(n_subjects):
        for i in range(per_subject):
            label = i % 2
            base = rng.normal(0, 0.1, size=8)
            if label:
                base[:4] += 2.0    # class 1：前四維明顯偏移
            X.append(base)
            y.append(label + 1)
            groups.append(s)
    return np.asarray(X), np.asarray(y), np.asarray(groups)


class TestLeakage:
    def test_groupkfold_subjects_never_overlap(self):
        X, y, groups = _synthetic_dataset()
        gkf = GroupKFold(n_splits=5)
        for tr, te in gkf.split(X, y, groups):
            assert set(groups[tr]).isdisjoint(set(groups[te]))

    def test_subject_cv_evaluate_runs_and_reports(self):
        X, y, groups = _synthetic_dataset(n_subjects=10, per_subject=8)
        report = subject_cv_evaluate(X, y, groups, kind="rf", n_splits=5)
        assert report["n_subjects"] == 10
        assert len(report["fold_accuracies"]) == 5
        # 合成資料可完美分離 → 各折 accuracy 應為 1.0
        assert report["mean_accuracy"] > 0.95


class TestModels:
    def test_make_model_rf(self):
        model = make_model("rf")
        assert hasattr(model, "n_estimators")

    def test_make_model_svm_is_scaled_pipeline(self):
        model = make_model("svm")
        assert hasattr(model, "named_steps")

    def test_make_model_unknown(self):
        import pytest
        with pytest.raises(ValueError):
            make_model("xgboost")
