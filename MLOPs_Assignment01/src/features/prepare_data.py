import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def prepare_data(data: pd.DataFrame, outcome: pd.DataFrame):
    """
    Prepares Heart Disease UCI dataset for machine learning.
    
    1. Binarizes the target 'num' (0 = Absence, 1 = Presence of Heart Disease).
    2. Imputes missing values (e.g. in 'ca' and 'thal') using median strategy.
    3. Scales numeric features using StandardScaler.
    4. Splits into train and test sets with stratification.
    """
    feature_columns = list(data.columns)
    
    X = data[feature_columns].copy()
    
    # Target in UCI Heart Disease: 0 is absence, 1-4 is presence -> binarize to 0/1
    if isinstance(outcome, pd.DataFrame):
        y_raw = outcome.iloc[:, 0].values
    elif isinstance(outcome, pd.Series):
        y_raw = outcome.values
    else:
        y_raw = np.array(outcome)
        
    y = (y_raw > 0).astype(int)

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
        transformers=[("numeric", preprocessing_pipeline, feature_columns)]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
