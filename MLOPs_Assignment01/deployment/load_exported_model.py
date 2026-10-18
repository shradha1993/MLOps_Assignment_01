import mlflow.pyfunc


def load_exported_model(
    model_path=r"C:\Users\shrad\MLOps_Assignment_01\MLOPs_Assignment01\deployment\model_export",
):
    model = mlflow.pyfunc.load_model(
        model_path
    )

    return model
