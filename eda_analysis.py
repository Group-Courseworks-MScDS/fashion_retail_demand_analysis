"""
eda.py

Exploratory Data Analysis for the Fashion Retail Demand Analysis project.

This module:
1. Loads the cleaned dataset
2. Generates descriptive statistics
3. Performs sales and pricing analysis
4. Examines categorical feature relationships
5. Produces visualizations for insights
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Configuration
# =========================
DATA_PATH = Path("data/cleaned/fashion_retail_cleaned.csv")
FIGURE_DIR = Path("reports/figures")

sns.set_style("whitegrid")


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
    """Ensure figure output directory exists."""
    directory.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load cleaned dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info("Dataset loaded: %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def descriptive_statistics(df: pd.DataFrame) -> None:
    """Print dataset descriptive statistics."""
    logger.info("Generating descriptive statistics")

    print("\n===== DESCRIPTIVE STATISTICS =====\n")
    print(df.describe())

    print("\n===== CATEGORICAL DISTRIBUTION =====\n")
    print(df.select_dtypes(include=["object", "string", "category"]).describe())


# =========================
# Visualization Functions
# =========================

def plot_price_distribution(df: pd.DataFrame) -> None:
    """Histogram of product prices."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], bins=20, kde=True)

    plt.title("Price Distribution")
    plt.xlabel("Price (USD)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "price_distribution.png")
    plt.close()

    logger.info("Saved: price_distribution.png")


def plot_sales_distribution(df: pd.DataFrame) -> None:
    """Histogram of sales volume."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df["sales_volume"], bins=20, kde=True)

    plt.title("Sales Volume Distribution")
    plt.xlabel("Sales Volume")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sales_distribution.png")
    plt.close()

    logger.info("Saved: sales_distribution.png")

def plot_sales_category_distribution(df: pd.DataFrame) -> None:
    """Bar plot of sales category distribution."""
    plt.figure(figsize=(6, 5))
    sns.countplot(x="sales_category", data=df)

    plt.title("Sales Category Distribution")
    plt.xlabel("Sales Category")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sales_category_distribution.png")
    plt.close()

    logger.info("Saved: sales_category_distribution.png")

def plot_promotion_vs_sales(df: pd.DataFrame) -> None:
    """Boxplot comparing promotion and sales."""
    plt.figure(figsize=(7, 5))

    sns.boxplot(
        x="promotion",
        y="sales_volume",
        data=df
    )

    plt.title("Promotion Impact on Sales")
    plt.xlabel("Promotion")
    plt.ylabel("Sales Volume")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "promotion_vs_sales.png")
    plt.close()

    logger.info("Saved: promotion_vs_sales.png")


def plot_season_vs_sales(df: pd.DataFrame) -> None:
    """Average sales by season."""
    season_sales = df.groupby("season")["sales_volume"].mean().sort_values()

    plt.figure(figsize=(8, 5))

    season_sales.plot(kind="bar")

    plt.title("Average Sales by Season")
    plt.xlabel("Season")
    plt.ylabel("Average Sales Volume")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sales_by_season.png")
    plt.close()

    logger.info("Saved: sales_by_season.png")


def plot_section_vs_sales(df: pd.DataFrame) -> None:
    """Average sales by section (MAN/WOMAN)."""
    section_sales = df.groupby("section")["sales_volume"].mean()

    plt.figure(figsize=(6, 5))

    section_sales.plot(kind="bar")

    plt.title("Average Sales by Section")
    plt.xlabel("Section")
    plt.ylabel("Average Sales Volume")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sales_by_section.png")
    plt.close()

    logger.info("Saved: sales_by_section.png")

def plot_price_vs_sales(df: pd.DataFrame) -> None:
    """Scatter plot for price vs sales."""
    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        x="price",
        y="sales_volume",
        hue="promotion",
        data=df
    )

    plt.title("Price vs Sales Volume")
    plt.xlabel("Price")
    plt.ylabel("Sales Volume")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "price_vs_sales.png")
    plt.close()

    logger.info("Saved: price_vs_sales.png")

def plot_price_category_vs_sales(df: pd.DataFrame) -> None:
    """Boxplot for price category vs sales."""
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x="price_category",
        y="sales_volume",
        data=df
    )

    plt.title("Price Category vs Sales Volume")
    plt.xlabel("Price Category")
    plt.ylabel("Sales Volume")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "price_category_vs_sales.png")
    plt.close()

    logger.info("Saved: price_category_vs_sales.png")

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap for numeric variables."""
    numeric_df = df.select_dtypes(include=["number"])

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png")
    plt.close()

    logger.info("Saved: correlation_heatmap.png")

def plot_top_10_best_selling_products(df: pd.DataFrame) -> None:
    """
    Plot the Top 10 Best-Selling Products based on sales volume.

    Args:
        df: Cleaned retail dataset.
    """

    top_products = (
        df.sort_values("sales_volume", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x="sales_volume",
        y="name",
        hue="name",          # assign hue
        data=top_products,
        palette="viridis",
    )

    plt.title("Top 10 Best-Selling Products")
    plt.xlabel("Sales Volume")
    plt.ylabel("Product Name")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "top_10_best_selling_products.png")
    plt.close()

    logger.info("Saved: top_10_best_selling_products.png")

def plot_product_position_sales(df: pd.DataFrame) -> None:
    """
    Analyze sales performance by product position in the store.
    """

    position_sales = (
        df.groupby("product_position")["sales_volume"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8,5))

    position_sales.plot(
        kind="bar",
        color="teal"
    )

    plt.title("Average Sales by Product Position")
    plt.xlabel("Product Position")
    plt.ylabel("Average Sales Volume")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "product_position_sales.png")
    plt.close()

    logger.info("Saved: product_position_sales.png")

def plot_material_sales(df: pd.DataFrame) -> None:
    """
    Analyze average sales performance by material type.
    """

    material_sales = (
        df.groupby("material")["sales_volume"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(8,5))

    material_sales.plot(
        kind="bar",
        color="purple"
    )

    plt.title("Top Materials by Average Sales")
    plt.xlabel("Material")
    plt.ylabel("Average Sales Volume")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "material_sales.png")
    plt.close()

    logger.info("Saved: material_sales.png")

def plot_origin_sales(df: pd.DataFrame) -> None:
    """
    Analyze average sales performance by product origin.
    """

    origin_sales = (
        df.groupby("origin")["sales_volume"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8,5))

    origin_sales.plot(
        kind="bar",
        color="orange"
    )

    plt.title("Average Sales by Product Origin")
    plt.xlabel("Origin")
    plt.ylabel("Average Sales Volume")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "origin_sales.png")
    plt.close()

    logger.info("Saved: origin_sales.png")

# =========================
# Main Pipeline
# =========================

def run_eda_pipeline() -> None:
    """Execute EDA workflow."""

    ensure_output_directory(FIGURE_DIR)

    df = load_dataset(DATA_PATH)

    descriptive_statistics(df)

    plot_price_distribution(df)
    plot_sales_distribution(df)
    plot_sales_category_distribution(df)
    plot_promotion_vs_sales(df)
    plot_season_vs_sales(df)
    plot_section_vs_sales(df)
    plot_price_vs_sales(df)
    plot_price_category_vs_sales(df)
    plot_correlation_heatmap(df)
    plot_top_10_best_selling_products(df)
    plot_product_position_sales(df)
    plot_material_sales(df)
    plot_origin_sales(df)
    logger.info("EDA pipeline completed successfully.")


if __name__ == "__main__":
    try:
        run_eda_pipeline()
    except Exception as e:
        print(f"Error during EDA pipeline: {e}")