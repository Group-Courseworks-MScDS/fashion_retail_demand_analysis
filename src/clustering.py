"""
clustering.py

Clustering analysis for the Fashion Retail Demand Analysis project.

This script:
1. Loads the cleaned dataset
2. Selects and prepares clustering features
3. Encodes categorical variables where needed
4. Scales numeric features
5. Uses the Elbow Method to inspect candidate cluster counts
6. Trains a K-Means model
7. Evaluates clustering using Silhouette Score
8. Saves clustered output and figures
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# =========================
# Configuration
# =========================
DATA_PATH: Final[Path] = Path("data/cleaned/fashion_retail_model_ready.csv")
OUTPUT_DIR: Final[Path] = Path("data/clustered")
FIGURE_DIR: Final[Path] = Path("reports/figures")

CLUSTERED_DATA_PATH: Final[Path] = OUTPUT_DIR / "fashion_retail_clustered.csv"
ELBOW_PLOT_PATH: Final[Path] = FIGURE_DIR / "kmeans_elbow_plot.png"
CLUSTER_PRICE_SALES_PLOT_PATH: Final[Path] = FIGURE_DIR / "cluster_price_sales_scatter.png"
CLUSTER_MEAN_SALES_PLOT_PATH: Final[Path] = FIGURE_DIR / "cluster_mean_sales.png"
PCA_PLOT_PATH: Final[Path] = FIGURE_DIR / "clusters_pca.png"
TSNE_PLOT_PATH: Final[Path] = FIGURE_DIR / "clusters_tsne.png"

DEFAULT_N_CLUSTERS: Final[int] = 4
RANDOM_STATE: Final[int] = 42

FEATURE_COLUMNS: Final[list[str]] = [
    "price",
    "sales_volume",
    "promotion_flag",
    "seasonal_flag",
    "product_position",
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
    Load cleaned dataset.

    Args:
        path: Path to CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded dataset with shape: %s", df.shape)
    return df


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Validate that required columns exist in the dataset.

    Args:
        df: Input DataFrame.
        required_columns: Columns required for clustering.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for clustering: {missing}")


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build preprocessing pipeline for clustering features.

    Args:
        numeric_features: Numeric feature names.
        categorical_features: Categorical feature names.

    Returns:
        Configured ColumnTransformer.
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Select clustering features and identify numeric/categorical groups.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of:
            - feature DataFrame
            - numeric feature names
            - categorical feature names
    """
    feature_df = df[FEATURE_COLUMNS].copy()

    numeric_features = [
        "price",
        "sales_volume",
        "promotion_flag",
        "seasonal_flag",
    ]

    categorical_features = [
        "product_position",
        "section",
        "season",
        "material",
        "origin",
    ]

    return feature_df, numeric_features, categorical_features


def generate_elbow_plot(
    transformed_features,
    min_k: int = 2,
    max_k: int = 8,
) -> None:
    """
    Generate and save Elbow Method plot.

    Args:
        transformed_features: Preprocessed feature matrix.
        min_k: Minimum number of clusters.
        max_k: Maximum number of clusters.
    """
    inertias: list[float] = []
    cluster_range = range(min_k, max_k + 1)

    for k in cluster_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        model.fit(transformed_features)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(cluster_range), inertias, marker="o")
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(ELBOW_PLOT_PATH)
    plt.close()

    logger.info("Saved elbow plot to: %s", ELBOW_PLOT_PATH)


def fit_kmeans(transformed_features, n_clusters: int) -> tuple[KMeans, float]:
    """
    Fit K-Means clustering model and compute silhouette score.

    Args:
        transformed_features: Preprocessed feature matrix.
        n_clusters: Number of clusters.

    Returns:
        Tuple of fitted model and silhouette score.
    """
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(transformed_features)
    score = silhouette_score(transformed_features, labels)

    logger.info("K-Means fitted with k=%s", n_clusters)
    logger.info("Silhouette Score: %.4f", score)

    return model, score


def add_cluster_labels(df: pd.DataFrame, labels) -> pd.DataFrame:
    """
    Add cluster labels to the original dataset.

    Args:
        df: Original DataFrame.
        labels: Cluster labels.

    Returns:
        DataFrame with cluster column.
    """
    df_clustered = df.copy()
    df_clustered["cluster"] = labels
    return df_clustered


def save_clustered_dataset(df: pd.DataFrame, path: Path) -> None:
    """
    Save clustered dataset.

    Args:
        df: Clustered DataFrame.
        path: Output CSV path.
    """
    df.to_csv(path, index=False)
    logger.info("Saved clustered dataset to: %s", path)


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple business-oriented cluster summary.

    Args:
        df: Clustered DataFrame.

    Returns:
        Summary DataFrame.
    """
    summary = (
        df.groupby("cluster")
        .agg(
            avg_price=("price", "mean"),
            avg_sales_volume=("sales_volume", "mean"),
            avg_revenue=("revenue_estimate", "mean"),
            promotion_rate=("promotion_flag", "mean"),
            seasonal_rate=("seasonal_flag", "mean"),
            product_count=("product_id", "count"),
        )
        .round(2)
        .reset_index()
    )

    logger.info("Cluster summary created.")
    print("\n===== CLUSTER SUMMARY =====\n")
    print(summary)

    return summary


def plot_cluster_price_vs_sales(df: pd.DataFrame) -> None:
    """
    Scatter plot of price vs sales volume colored by cluster.

    Args:
        df: Clustered DataFrame.
    """
    plt.figure(figsize=(8, 5))

    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        plt.scatter(
            subset["price"],
            subset["sales_volume"],
            label=f"Cluster {cluster_id}",
            alpha=0.7,
        )

    plt.title("Clusters by Price and Sales Volume")
    plt.xlabel("Price")
    plt.ylabel("Sales Volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CLUSTER_PRICE_SALES_PLOT_PATH)
    plt.close()

    logger.info("Saved cluster scatter plot to: %s", CLUSTER_PRICE_SALES_PLOT_PATH)


def plot_cluster_mean_sales(summary_df: pd.DataFrame) -> None:
    """
    Bar chart of average sales volume by cluster.

    Args:
        summary_df: Cluster summary DataFrame.
    """
    plt.figure(figsize=(7, 5))
    plt.bar(summary_df["cluster"].astype(str), summary_df["avg_sales_volume"])
    plt.title("Average Sales Volume by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Average Sales Volume")
    plt.tight_layout()
    plt.savefig(CLUSTER_MEAN_SALES_PLOT_PATH)
    plt.close()

    logger.info("Saved cluster mean sales plot to: %s", CLUSTER_MEAN_SALES_PLOT_PATH)


def plot_clusters_pca(transformed_features, labels) -> None:
    """
    Visualize clusters using PCA (2D).

    Args:
        transformed_features: Preprocessed feature matrix
        labels: Cluster labels
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(transformed_features)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        alpha=0.6,
        s=15
    )

    plt.title("Clusters Visualization (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(PCA_PLOT_PATH)
    plt.close()

    logger.info("Saved PCA cluster plot to: %s", PCA_PLOT_PATH)


def plot_clusters_tsne(transformed_features, labels) -> None:
    """
    Visualize clusters using t-SNE (2D).

    Args:
        transformed_features: Preprocessed feature matrix
        labels: Cluster labels
    """
    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        perplexity=30,
        max_iter=1000,
        learning_rate="auto",
        init="pca"
    )

    reduced = tsne.fit_transform(transformed_features)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        alpha=0.6,
        s=15
    )

    plt.title("Clusters Visualization (t-SNE)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(TSNE_PLOT_PATH)
    plt.close()

    logger.info("Saved t-SNE cluster plot to: %s", TSNE_PLOT_PATH)


def run_clustering_pipeline(n_clusters: int = DEFAULT_N_CLUSTERS) -> None:
    """
    Execute full clustering workflow.

    Args:
        n_clusters: Final number of clusters for K-Means.
    """
    ensure_directories()

    df = load_dataset(DATA_PATH)
    validate_required_columns(df, FEATURE_COLUMNS + ["product_id", "revenue_estimate"])

    feature_df, numeric_features, categorical_features = prepare_features(df)

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    transformed_features = preprocessor.fit_transform(feature_df)
    logger.info("Feature preprocessing completed.")

    generate_elbow_plot(transformed_features)

    model, silhouette = fit_kmeans(transformed_features, n_clusters=n_clusters)
    labels = model.labels_

    df_clustered = add_cluster_labels(df, labels)
    save_clustered_dataset(df_clustered, CLUSTERED_DATA_PATH)

    summary_df = summarize_clusters(df_clustered)

    plot_cluster_price_vs_sales(df_clustered)
    plot_cluster_mean_sales(summary_df)
    plot_clusters_pca(transformed_features, labels)
    
    """
    The dataset is large (>10k rows), to reduce t-SNE slowness
    """
    sample_size = min(3000, transformed_features.shape[0])
    indices = np.random.choice(transformed_features.shape[0], sample_size, replace=False)
    sampled_features = transformed_features[indices]
    sampled_labels = labels[indices]
    plot_clusters_tsne(sampled_features, sampled_labels)

    logger.info("Clustering pipeline completed successfully.")
    logger.info("Final silhouette score: %.4f", silhouette)


if __name__ == "__main__":
    try:
        run_clustering_pipeline()
    except Exception as e:
        print(f"Error during clustering pipeline: {e}")