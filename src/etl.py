"""
etl.py

Extract, Load, and Transform pipeline for the Fashion Retail Demand Analysis project.

This script:
1. Loads the raw dataset
2. Cleans and standardizes the data
3. Handles missing values and duplicates
4. Caps outliers in selected numeric columns
5. Creates engineered features
6. Saves cleaned datasets for analysis and modeling
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd


# =========================
# Configuration
# =========================
RAW_DATA_PATH = Path("data/raw/fashion_retail.csv")
CLEANED_DATA_DIR = Path("data/cleaned")
CLEANED_DATA_PATH = CLEANED_DATA_DIR / "fashion_retail_cleaned.csv"
MODEL_READY_DATA_PATH = CLEANED_DATA_DIR / "fashion_retail_model_ready.csv"

NUMERIC_COLUMNS = ["product_id", "sales_volume", "price"]
YES_NO_COLUMNS = ["promotion", "seasonal"]

TEXT_COLUMNS = [
    "product_position",
    "promotion",
    "product_category",
    "seasonal",
    "brand",
    "name",
    "description",
    "currency",
    "terms",
    "section",
    "season",
    "material",
    "origin",
]

TITLE_CASE_COLUMNS = [
    "product_position",
    "product_category",
    "brand",
    "terms",
    "season",
    "material",
    "origin",
]

OUTLIER_COLUMNS = ["price", "sales_volume"]


# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =========================
# Utility Functions
# =========================
def ensure_output_directory(directory: Path) -> None:
    """Create the output directory if it does not exist."""
    directory.mkdir(parents=True, exist_ok=True)


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV.

    Args:
        file_path: Path to the raw CSV file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)
    logger.info("Dataset loaded successfully: %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with standardized column names.
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    logger.info("Column names standardized.")
    return df


def clean_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Trim whitespace in text columns.

    Args:
        df: Input DataFrame.
        columns: Columns to clean.

    Returns:
        Updated DataFrame.
    """
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()
    logger.info("Text columns cleaned.")
    return df


def convert_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Convert specified columns to numeric.

    Invalid values become NaN.

    Args:
        df: Input DataFrame.
        columns: Numeric columns to convert.

    Returns:
        Updated DataFrame.
    """
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    logger.info("Numeric columns converted.")
    return df


def standardize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize formatting for selected categorical columns.

    Args:
        df: Input DataFrame.

    Returns:
        Updated DataFrame.
    """
    df = df.copy()

    for column in YES_NO_COLUMNS:
        if column in df.columns:
            df[column] = df[column].str.title()

    for column in TITLE_CASE_COLUMNS:
        if column in df.columns:
            df[column] = df[column].str.title()

    if "section" in df.columns:
        df["section"] = df["section"].str.upper()

    if "currency" in df.columns:
        df["currency"] = df["currency"].str.upper()

    logger.info("Categorical values standardized.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame without duplicates.
    """
    initial_rows = len(df)
    df = df.drop_duplicates().copy()
    removed_rows = initial_rows - len(df)
    logger.info("Removed %s duplicate rows.", removed_rows)
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values.

    Strategy:
    - Numeric columns: median
    - Categorical columns: mode

    Args:
        df: Input DataFrame.

    Returns:
        Updated DataFrame.
    """
    df = df.copy()

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    for column in numeric_columns:
        if df[column].isna().any():
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)

    for column in categorical_columns:
        if df[column].isna().any():
            mode_series = df[column].mode(dropna=True)
            if not mode_series.empty:
                df[column] = df[column].fillna(mode_series.iloc[0])

    logger.info("Missing values handled.")
    return df


def filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid numeric records.

    Rules:
    - price must be > 0
    - sales_volume must be >= 0

    Args:
        df: Input DataFrame.

    Returns:
        Filtered DataFrame.
    """
    df = df.copy()
    initial_rows = len(df)

    if "price" in df.columns:
        df = df[df["price"] > 0]

    if "sales_volume" in df.columns:
        df = df[df["sales_volume"] >= 0]

    removed_rows = initial_rows - len(df)
    logger.info("Removed %s invalid rows.", removed_rows)
    return df


def cap_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Cap outliers in a numeric column using the IQR method.

    Args:
        df: Input DataFrame.
        column: Column to cap.

    Returns:
        Updated DataFrame.
    """
    df = df.copy()

    if column not in df.columns:
        return df

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    logger.info(
        "Outliers capped for '%s' using IQR bounds [%.2f, %.2f].",
        column,
        lower_bound,
        upper_bound,
    )
    return df


def cap_selected_outliers(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Apply IQR outlier capping to selected columns.

    Args:
        df: Input DataFrame.
        columns: List of numeric columns.

    Returns:
        Updated DataFrame.
    """
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df = cap_outliers_iqr(df, column)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features for downstream analysis and modeling.

    Features:
    - promotion_flag
    - seasonal_flag
    - revenue_estimate
    - name_length
    - description_length
    - price_category
    - sales_category

    Args:
        df: Input DataFrame.

    Returns:
        Updated DataFrame.
    """
    df = df.copy()

    if "promotion" in df.columns:
        df["promotion_flag"] = df["promotion"].map({"Yes": 1, "No": 0})

    if "seasonal" in df.columns:
        df["seasonal_flag"] = df["seasonal"].map({"Yes": 1, "No": 0})

    if {"price", "sales_volume"}.issubset(df.columns):
        df["revenue_estimate"] = df["price"] * df["sales_volume"]

    if "name" in df.columns:
        df["name_length"] = df["name"].astype("string").str.len()

    if "description" in df.columns:
        df["description_length"] = df["description"].astype("string").str.len()

    if "price" in df.columns and df["price"].nunique() >= 3:
        df["price_category"] = pd.qcut(
            df["price"],
            q=3,
            labels=["Low", "Medium", "High"],
            duplicates="drop",
        )

    if "sales_volume" in df.columns and df["sales_volume"].nunique() >= 3:
        df["sales_category"] = pd.qcut(
            df["sales_volume"],
            q=3,
            labels=["Low", "Medium", "High"],
            duplicates="drop",
        )

    logger.info("Engineered features added.")
    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that have only one unique value.

    Args:
        df: Input DataFrame.

    Returns:
        Updated DataFrame.
    """
    df = df.copy()
    constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_columns:
        df = df.drop(columns=constant_columns)
        logger.info("Dropped constant columns: %s", constant_columns)
    return df


def create_model_ready_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a modeling-friendly version of the cleaned dataset.

    Drops columns that are unlikely to help structured modeling.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Model-ready DataFrame.
    """
    df_model = df.copy()

    columns_to_drop = [column for column in ["url"] if column in df_model.columns]
    if columns_to_drop:
        df_model = df_model.drop(columns=columns_to_drop)

    logger.info("Model-ready dataset created.")
    return df_model


def log_data_quality_summary(df: pd.DataFrame, stage_name: str) -> None:
    """
    Log a short quality summary of the dataset.

    Args:
        df: DataFrame to inspect.
        stage_name: Name of the processing stage.
    """
    logger.info("=== %s ===", stage_name)
    logger.info("Shape: %s rows, %s columns", df.shape[0], df.shape[1])
    logger.info("Missing values:\n%s", df.isna().sum())
    logger.info("Data types:\n%s", df.dtypes)


def save_dataset(df: pd.DataFrame, file_path: Path) -> None:
    """
    Save DataFrame to CSV.

    Args:
        df: DataFrame to save.
        file_path: Output path.
    """
    df.to_csv(file_path, index=False)
    logger.info("Saved dataset to: %s", file_path)


def run_etl_pipeline() -> None:
    """Execute the full ETL pipeline."""
    ensure_output_directory(CLEANED_DATA_DIR)

    df = load_dataset(RAW_DATA_PATH)
    log_data_quality_summary(df, "Raw Dataset")

    df = standardize_column_names(df)
    df = clean_text_columns(df, TEXT_COLUMNS)
    df = convert_numeric_columns(df, NUMERIC_COLUMNS)
    df = standardize_categorical_values(df)
    df = remove_duplicates(df)
    df = fill_missing_values(df)
    df = filter_invalid_rows(df)
    df = cap_selected_outliers(df, OUTLIER_COLUMNS)
    df = add_engineered_features(df)
    df = drop_constant_columns(df)

    log_data_quality_summary(df, "Cleaned Dataset")

    df_model = create_model_ready_dataset(df)

    save_dataset(df, CLEANED_DATA_PATH)
    save_dataset(df_model, MODEL_READY_DATA_PATH)

    logger.info("ETL pipeline completed successfully.")


if __name__ == "__main__":
    try:
        run_etl_pipeline()
    except Exception as e:
        print(f"Error during ETL pipeline: {e}")