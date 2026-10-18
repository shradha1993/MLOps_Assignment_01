def predict_with_packaged_model(
    model,
    X,
):
    predictions = model.predict(X)

    print(
        f"Predictions generated: "
        f"{len(predictions)}"
    )

    return predictions
