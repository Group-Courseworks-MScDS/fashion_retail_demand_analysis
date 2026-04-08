from src.etl import run_etl_pipeline
from src.eda_analysis import run_eda_pipeline
from src.clustering import run_clustering_pipeline
from src.prediction import run_prediction_pipeline

def main() -> None:
    run_etl_pipeline()
    run_eda_pipeline()
    run_clustering_pipeline()
    run_prediction_pipeline()
    
if __name__ == "__main__":
    main()