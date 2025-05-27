"""
Test script for ML model builder tool
"""

import pandas as pd
from pathlib import Path
from server.tools.ml_model_builder import MLModelBuilder

# Initialize ML builder
data_path = Path("./data/synthetic")
ml_builder = MLModelBuilder(data_path)

# Load sample data
print("Loading synthetic data...")
patient_df = pd.read_csv(data_path / "patient.csv")
hospitalization_df = pd.read_csv(data_path / "hospitalization.csv")

# Merge for basic dataset
df = patient_df.merge(hospitalization_df, on="patient_id")

# Create a simple mortality outcome (synthetic)
import numpy as np
np.random.seed(42)
df['mortality'] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])

# Define features
features = ['age', 'admission_year']

print(f"\nDataset shape: {df.shape}")
print(f"Outcome distribution: {df['mortality'].value_counts()}")
print(f"Features: {features}")

# Build model
print("\nBuilding ML model...")
result = ml_builder.build_model(
    df=df,
    outcome='mortality',
    features=features,
    model_type='auto',
    hyperparameter_tuning=False,  # Disable for speed
    validation_split=0.2
)

print("\nModel Results:")
print(f"Model ID: {result['model_id']}")
print(f"Model Type: {result['model_type']}")
print(f"Task: {result['task']}")
print(f"Number of samples: {result['n_samples']}")
print(f"Number of features: {result['n_features']}")

print("\nPerformance Metrics:")
print("Training:")
for metric, value in result['performance']['train'].items():
    print(f"  - {metric}: {value:.3f}")

print("\nTest:")
for metric, value in result['performance']['test'].items():
    print(f"  - {metric}: {value:.3f}")

if result['feature_importance']:
    print("\nFeature Importance:")
    for feature, importance in result['feature_importance'].items():
        print(f"  - {feature}: {importance:.3f}")

# Generate model report
print("\nGenerating model report...")
report = ml_builder.generate_model_report(result['model_id'])
print(report)

print("\nML tool test completed successfully!")