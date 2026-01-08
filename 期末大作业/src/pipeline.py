import argparse
from .clean import clean_data
from .eda import run_eda_analysis
from .cluster import run_clustering_pipeline
from .label import run_labeling_pipeline
from .metrics_lifecycle import run_lifecycle_analysis
from .metrics_inequality import run_inequality_analysis
from .model import run_statistical_modeling
from .robustness import run_robustness_checks
from .report import generate_report

def main():
    parser = argparse.ArgumentParser(description="COVID Tweets Analysis Pipeline")
    parser.add_argument("step", choices=["clean", "eda", "cluster", "label", "metrics", "inequality", "model", "robustness", "report"], help="Pipeline step to execute")
    
    args = parser.parse_args()
    
    if args.step == "clean":
        clean_data()
    elif args.step == "eda":
        run_eda_analysis()
    elif args.step == "cluster":
        run_clustering_pipeline()
    elif args.step == "label":
        run_labeling_pipeline()
    elif args.step == "metrics":
        run_lifecycle_analysis()
    elif args.step == "inequality":
        run_inequality_analysis()
    elif args.step == "model":
        run_statistical_modeling()
    elif args.step == "robustness":
        run_robustness_checks()
    elif args.step == "report":
        generate_report()

if __name__ == "__main__":
    main()
