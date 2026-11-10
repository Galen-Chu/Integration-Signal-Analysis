"""Model training · RF/SVM＋subject-wise 交叉驗證（防受試者洩漏）.

隨機切分會讓同一受試者的視窗同時落在訓練與測試——accuracy 虛高。
本專案一律 GroupKFold（groups=受試者編號），這是結果可信的根基.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class _ZeroBasedWrapper(BaseEstimator, ClassifierMixin):
    """把任意整數標籤映射到 0 起始再交給內部分類器（XGBoost 的要求）.

    UCI HAR 活動代碼 1-6、PAMAP2 1-24——直接餵 XGBClassifier 會拒絕.
    """

    def __init__(self, base):
        self.base = base

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.base.fit(X, np.searchsorted(self.classes_, y))
        return self

    def predict(self, X):
        return self.classes_[self.base.predict(X)]


def make_model(kind: str = "rf", random_state: int = 42):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=random_state)
    if kind == "svm":
        return Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, random_state=random_state)),
        ])
    if kind == "xgb":
        from xgboost import XGBClassifier  # 延遲匯入：無 xgboost 環境仍可用其餘模型

        return _ZeroBasedWrapper(Pipeline([
            ("scale", StandardScaler()),
            ("xgb", XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.2,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="mlogloss", random_state=random_state,
                n_jobs=-1)),
        ]))
    raise ValueError(f"unknown model kind: {kind}")


def subject_cv_evaluate(X, y, groups, kind: str = "rf", n_splits: int = 5,
                        random_state: int = 42) -> dict:
    """GroupKFold 逐折訓練/評估；回傳彙整報告（含混淆矩陣總和）."""
    labels = np.unique(y)
    gkf = GroupKFold(n_splits=n_splits)
    accs, all_true, all_pred = [], [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        assert set(groups[tr]).isdisjoint(set(groups[te])), \
            f"fold {fold}: subject leakage!"
        model = make_model(kind, random_state)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        accs.append(float((pred == y[te]).mean()))
        all_true.extend(y[te].tolist())
        all_pred.extend(pred.tolist())
        print(f"  fold {fold}/{n_splits} accuracy = {accs[-1]:.4f} "
              f"(test subjects: {sorted(set(groups[te].tolist()))})")

    from sklearn.metrics import confusion_matrix, f1_score
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    return {
        "model": kind,
        "n_splits": n_splits,
        "fold_accuracies": accs,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "macro_f1": float(f1_score(all_true, all_pred, average="macro")),
        "confusion_matrix": cm.tolist(),
        "labels": labels.tolist(),
        "n_train_windows": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
    }


def evaluate_holdout(X_train, y_train, X_test, y_test, groups_train,
                     kind: str = "rf", random_state: int = 42) -> dict:
    """官方 held-out 評估：訓練集全量擬合、測試集一次評估.

    UCI HAR 的官方 train/test 本身就是受試者不重疊（21/9 人）——
    此模式可與文獻在相同條件下的數字直接對照.
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score

    # 受試者 disjoint 由呼叫端保證（UCI 官方切分本身即 subject-wise）
    model = make_model(kind, random_state)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = np.unique(np.concatenate([y_train, y_test]))
    cm = confusion_matrix(y_test, pred, labels=labels)
    return {
        "model": kind,
        "split": "official-holdout",
        "accuracy": float((pred == y_test).mean()),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "confusion_matrix": cm.tolist(),
        "labels": labels.tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
