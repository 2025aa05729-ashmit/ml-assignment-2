"""
Model Evaluation Script
This script trains all classification models and prints evaluation metrics.
"""

from model.train_models import train_all_models

def main():
    models, metrics_df, _ = train_all_models()
    print("Model Evaluation Results:")
    print(metrics_df)

if __name__ == "__main__":
    main()
