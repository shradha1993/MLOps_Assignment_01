from src.models.model_pipelines import get_models


def train_models(X_train, y_train):
    models = get_models()
    trained_models = {}

    for name, model in models.items():
        print(f"Training: {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models
