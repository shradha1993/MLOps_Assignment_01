from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def evaluate_models(trained_models, X_test, y_test):
    results = {}

    for name, model in trained_models.items():
        predictions = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="macro", zero_division=0),
            "recall": recall_score(y_test, predictions, average="macro", zero_division=0),
            "f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        }

    return results
