"""Model training · RF/SVM＋subject-wise 交叉驗證（防受試者洩漏）.

隨機切分會讓同一受試者的視窗同時落在訓練與測試——accuracy 虛高。
本專案一律 GroupKFold（groups=受試者編號），這是結果可信的根基.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_model(kind: str = "rf", random_state: int = 42):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=random_state)
    if kind == "svm":
        return Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, random_state=random_state)),
        ])
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
