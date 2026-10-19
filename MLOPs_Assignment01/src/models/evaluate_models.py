from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluates trained classification models on test dataset.
    Calculates Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    """
    results = {}

    for name, model in trained_models.items():
        predictions = model.predict(X_test)

        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                y_prob = predictions
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            roc_auc = float(accuracy_score(y_test, predictions))

        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, average="binary", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, average="binary", zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, average="binary", zero_division=0)), 4),
            "roc_auc": round(roc_auc, 4),
        }

    return results
