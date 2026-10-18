import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def prepare_data(data, outcome):
    FEATURE_COLUMNS = list(data.columns)
    TARGET_COLUMN = list(outcome.columns)
    #ZERO_AS_MISSING_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMN

    X = data[FEATURE_COLUMNS].copy()
    y = outcome[TARGET_COLUMN].copy()

    X[FEATURE_COLUMNS] = X[FEATURE_COLUMNS].replace(0, np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessing_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("numeric", preprocessing_pipeline, FEATURE_COLUMNS)]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
