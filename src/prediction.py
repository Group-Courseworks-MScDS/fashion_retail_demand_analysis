"""
prediction.py

Predictive modeling module for the Fashion Retail Demand Analysis project.

This script:
1. Loads the cleaned model-ready dataset
2. Selects features and target
3. Builds preprocessing + regression pipelines
4. Trains and evaluates multiple models
5. Saves predictions, metrics, and feature importance outputs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# Configuration
# =========================
DATA_PATH: Final[Path] = Path("data/cleaned/fashion_retail_model_ready.csv")
OUTPUT_DIR: Final[Path] = Path("data/predictions")
FIGURE_DIR: Final[Path] = Path("reports/figures")
METRICS_PATH: Final[Path] = OUTPUT_DIR / "model_metrics.json"
PREDICTIONS_PATH: Final[Path] = OUTPUT_DIR / "sales_volume_predictions.csv"
FEATURE_IMPORTANCE_PATH: Final[Path] = OUTPUT_DIR / "random_forest_feature_importance.csv"
FUTURE_FORECAST_PATH: Final[Path] = OUTPUT_DIR / "future_demand_forecasts.csv"

ACTUAL_VS_PREDICTED_PLOT_PATH: Final[Path] = FIGURE_DIR / "actual_vs_predicted_sales.png"
RESIDUAL_PLOT_PATH: Final[Path] = FIGURE_DIR / "residual_plot_random_forest.png"
FEATURE_IMPORTANCE_PLOT_PATH: Final[Path] = FIGURE_DIR / "feature_importance_random_forest.png"
FORECAST_PLOT_PATH: Final[Path] = FIGURE_DIR / "future_demand_forecast.png"

RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.20

TARGET_COLUMN: Final[str] = "sales_volume"

FEATURE_COLUMNS: Final[list[str]] = [
    "price",
    "promotion_flag",
    "seasonal_flag",
    "product_position",
    "terms",
    "section",
    "season",
    "material",
    "origin",
]


# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =========================
# Helpers
# =========================
def ensure_directories() -> None:
    """Create required output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV.

    Args:
        path: File path to the dataset.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded dataset with shape: %s", df.shape)
    return df


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Validate that all required columns are present.

    Args:
        df: Input DataFrame.
        required_columns: Required column names.

    Raises:
        ValueError: If columns are missing.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def split_features_target(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features and target.

    Args:
        df: Input DataFrame.
        feature_columns: Feature column names.
        target_column: Target column name.

    Returns:
        Feature matrix X and target vector y.
    """
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    return X, y


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build preprocessing transformer.

    Args:
        numeric_features: Numeric columns.
        categorical_features: Categorical columns.

    Returns:
        ColumnTransformer object.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """
    Build model pipelines.

    Args:
        preprocessor: Shared preprocessing transformer.

    Returns:
        Dictionary of model pipelines.
    """
    models = {
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                )),
            ]
        ),
    }
    return models


def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


def train_and_evaluate_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, dict[str, float]], dict[str, pd.Series]]:
    """
    Train and evaluate all candidate models.

    Args:
        models: Model pipelines.
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.

    Returns:
        Metrics by model and predictions by model.
    """
    metrics_by_model: dict[str, dict[str, float]] = {}
    predictions_by_model: dict[str, pd.Series] = {}

    for model_name, pipeline in models.items():
        logger.info("Training model: %s", model_name)
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, predictions)

        metrics_by_model[model_name] = metrics
        predictions_by_model[model_name] = pd.Series(predictions, index=y_test.index)

        logger.info("%s metrics: %s", model_name, metrics)

    return metrics_by_model, predictions_by_model


def select_best_model(metrics_by_model: dict[str, dict[str, float]]) -> str:
    """
    Select best model based on lowest RMSE.

    Args:
        metrics_by_model: Metrics dictionary.

    Returns:
        Name of best model.
    """
    best_model_name = min(
        metrics_by_model,
        key=lambda model_name: metrics_by_model[model_name]["RMSE"],
    )
    logger.info("Best model selected: %s", best_model_name)
    return best_model_name


def save_metrics(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    """
    Save model metrics to JSON.

    Args:
        metrics: Metrics dictionary.
        output_path: Output JSON path.
    """
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    logger.info("Saved metrics to: %s", output_path)



def run_prediction_pipeline() -> None:
    """Execute full predictive modeling workflow."""
    ensure_directories()

    df = load_dataset(DATA_PATH)
    validate_required_columns(df, FEATURE_COLUMNS + [TARGET_COLUMN])

    X, y = split_features_target(df, FEATURE_COLUMNS, TARGET_COLUMN)

    numeric_features = ["price", "promotion_flag", "seasonal_flag"]
    categorical_features = [
        "product_position",
        "terms",
        "section",
        "season",
        "material",
        "origin",
    ]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    logger.info(
        "Train/Test split completed. Train: %s rows, Test: %s rows",
        X_train.shape[0],
        X_test.shape[0],
    )

    models = build_models(preprocessor)
    metrics_by_model, predictions_by_model = train_and_evaluate_models(
        models=models,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    save_metrics(metrics_by_model, METRICS_PATH)

    best_model_name = select_best_model(metrics_by_model)
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    best_predictions = pd.Series(best_model.predict(X_test), index=y_test.index)

    results_df = save_predictions(
        X_test=X_test,
        y_test=y_test,
        y_pred=best_predictions,
        output_path=PREDICTIONS_PATH,
    )

    

if __name__ == "__main__":
    try:
        run_prediction_pipeline()
    except Exception as e:
        print(f"Error during prediction pipeline: {e}")