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


def save_predictions(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    """
    Save actual vs predicted values.

    Args:
        X_test: Test feature set.
        y_test: Actual target values.
        y_pred: Predicted target values.
        output_path: CSV output path.

    Returns:
        Output DataFrame with predictions.
    """
    results_df = X_test.copy()
    results_df["actual_sales_volume"] = y_test
    results_df["predicted_sales_volume"] = y_pred.round(2)
    results_df["residual"] = results_df["actual_sales_volume"] - results_df["predicted_sales_volume"]

    results_df.to_csv(output_path, index=False)
    logger.info("Saved predictions to: %s", output_path)

    return results_df


def plot_actual_vs_predicted(results_df: pd.DataFrame) -> None:
    """
    Plot actual vs predicted sales volume.

    Args:
        results_df: DataFrame containing actual and predicted values.
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(
        results_df["actual_sales_volume"],
        results_df["predicted_sales_volume"],
        alpha=0.7,
    )

    min_value = min(
        results_df["actual_sales_volume"].min(),
        results_df["predicted_sales_volume"].min(),
    )
    max_value = max(
        results_df["actual_sales_volume"].max(),
        results_df["predicted_sales_volume"].max(),
    )
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.title("Actual vs Predicted Sales Volume")
    plt.xlabel("Actual Sales Volume")
    plt.ylabel("Predicted Sales Volume")
    plt.tight_layout()
    plt.savefig(ACTUAL_VS_PREDICTED_PLOT_PATH)
    plt.close()

    logger.info("Saved actual vs predicted plot to: %s", ACTUAL_VS_PREDICTED_PLOT_PATH)


def plot_residuals(results_df: pd.DataFrame) -> None:
    """
    Plot residuals against predicted values.

    Args:
        results_df: DataFrame containing predictions and residuals.
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(
        results_df["predicted_sales_volume"],
        results_df["residual"],
        alpha=0.7,
    )
    plt.axhline(y=0, linestyle="--")

    plt.title("Residual Plot")
    plt.xlabel("Predicted Sales Volume")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(RESIDUAL_PLOT_PATH)
    plt.close()

    logger.info("Saved residual plot to: %s", RESIDUAL_PLOT_PATH)


def extract_feature_importance(
    model_pipeline: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """
    Extract feature importance from trained random forest model.

    Args:
        model_pipeline: Trained pipeline.
        numeric_features: Numeric feature names.
        categorical_features: Categorical feature names.

    Returns:
        Feature importance DataFrame.
    """
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    encoded_feature_names = preprocessor.get_feature_names_out()
    feature_importance_df = pd.DataFrame({
        "feature": encoded_feature_names,
        "importance": model.feature_importances_,
    }).sort_values(by="importance", ascending=False)

    return feature_importance_df


def save_and_plot_feature_importance(feature_importance_df: pd.DataFrame) -> None:
    """
    Save feature importance to CSV and plot top features.

    Args:
        feature_importance_df: Feature importance DataFrame.
    """
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    logger.info("Saved feature importance to: %s", FEATURE_IMPORTANCE_PATH)

    top_features = feature_importance_df.head(10).iloc[::-1]

    plt.figure(figsize=(8, 5))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH)
    plt.close()

    logger.info("Saved feature importance plot to: %s", FEATURE_IMPORTANCE_PLOT_PATH)


def create_future_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create realistic future business scenarios for demand forecasting.
    These are not historical time points. They represent upcoming retail conditions.
    """
    median_price = df["price"].median()
    common_position = df["product_position"].mode()[0]
    common_terms = df["terms"].mode()[0]
    common_material = df["material"].mode()[0]
    common_origin = df["origin"].mode()[0]

    scenarios = pd.DataFrame([
        {
            "scenario_name": "Spring promo - WOMAN",
            "price": median_price,
            "promotion_flag": 1,
            "seasonal_flag": 1,
            "product_position": common_position,
            "terms": common_terms,
            "section": "WOMAN",
            "season": "Spring",
            "material": common_material,
            "origin": common_origin,
        },
        {
            "scenario_name": "Summer no promo - MAN",
            "price": median_price,
            "promotion_flag": 0,
            "seasonal_flag": 1,
            "product_position": common_position,
            "terms": common_terms,
            "section": "MAN",
            "season": "Summer",
            "material": common_material,
            "origin": common_origin,
        },
        {
            "scenario_name": "Autumn promo - WOMAN",
            "price": median_price,
            "promotion_flag": 1,
            "seasonal_flag": 1,
            "product_position": common_position,
            "terms": common_terms,
            "section": "WOMAN",
            "season": "Autumn",
            "material": common_material,
            "origin": common_origin,
        },
        {
            "scenario_name": "Winter no promo - MAN",
            "price": median_price,
            "promotion_flag": 0,
            "seasonal_flag": 1,
            "product_position": common_position,
            "terms": common_terms,
            "section": "MAN",
            "season": "Winter",
            "material": common_material,
            "origin": common_origin,
        },
    ])

    return scenarios


def save_future_forecasts(best_model: Pipeline, scenario_df: pd.DataFrame) -> pd.DataFrame:
    forecast_input = scenario_df[FEATURE_COLUMNS].copy()
    forecast_values = best_model.predict(forecast_input)

    forecast_df = scenario_df.copy()
    forecast_df["forecast_sales_volume"] = forecast_values.round(2)
    forecast_df.to_csv(FUTURE_FORECAST_PATH, index=False)

    return forecast_df


def plot_future_forecasts(forecast_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(forecast_df["scenario_name"], forecast_df["forecast_sales_volume"])
    plt.title("Forecasted Demand for Future Retail Scenarios")
    plt.xlabel("Scenario")
    plt.ylabel("Forecast Sales Volume")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FORECAST_PLOT_PATH)
    plt.close()


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

    plot_actual_vs_predicted(results_df)
    plot_residuals(results_df)

    if best_model_name == "random_forest":
        feature_importance_df = extract_feature_importance(
            model_pipeline=best_model,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        save_and_plot_feature_importance(feature_importance_df)

    future_scenarios = create_future_scenarios(df)
    forecast_df = save_future_forecasts(best_model, future_scenarios)
    plot_future_forecasts(forecast_df)

    logger.info("Demand forecasting pipeline completed successfully.")


if __name__ == "__main__":
    try:
        run_prediction_pipeline()
    except Exception as e:
        print(f"Error during prediction pipeline: {e}")