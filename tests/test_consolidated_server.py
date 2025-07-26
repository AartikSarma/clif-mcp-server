#!/usr/bin/env python
"""
Test script for consolidated CLIF MCP Server
Tests ML model builder functionality
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from clif_server import CLIFServer


async def test_ml_functionality():
    """Test ML model builder in consolidated server"""
    
    print("Testing Consolidated CLIF Server ML Functionality")
    print("=" * 50)
    
    # Initialize server
    data_path = "./data/synthetic"
    server = CLIFServer(data_path)
    
    print(f"\n1. Server initialized with data path: {data_path}")
    print(f"   Available tables: {server.data_explorer.tables}")
    
    # Test 1: Build a simple cohort
    print("\n2. Building test cohort...")
    criteria = {
        "age_range": {"min": 18, "max": 100},
        "exclude_readmissions": True
    }
    
    cohort_df, cohort_id = server.cohort_builder.build_cohort(criteria)
    print(f"   Cohort built: {len(cohort_df)} hospitalizations")
    
    # Test 2: Create derived outcome if needed
    print("\n3. Preparing data for ML...")
    if "mortality" not in cohort_df.columns:
        cohort_df["mortality"] = (cohort_df["discharge_disposition"] == "Expired").astype(int)
        print("   Created mortality outcome from discharge_disposition")
    
    print(f"   Mortality rate: {cohort_df['mortality'].mean():.2%}")
    
    # Test 3: Build ML model
    print("\n4. Building ML model...")
    
    # Simple features
    features = ["age", "admission_year"]
    
    # Check if features exist
    missing_features = [f for f in features if f not in cohort_df.columns]
    if missing_features:
        print(f"   Warning: Missing features {missing_features}")
        # Use available numeric columns
        numeric_cols = cohort_df.select_dtypes(include=['int64', 'float64']).columns
        features = [col for col in numeric_cols if col not in ['mortality', 'patient_id', 'hospitalization_id']][:2]
        print(f"   Using available features: {features}")
    
    try:
        result = server.ml_model_builder.build_model(
            df=cohort_df,
            outcome='mortality',
            features=features,
            model_type='auto',
            hyperparameter_tuning=False,  # Disable for speed
            validation_split=0.2
        )
        
        print("\n5. Model Results:")
        print(f"   Model ID: {result['model_id']}")
        print(f"   Model Type: {result['model_type']}")
        print(f"   Task: {result['task']}")
        print(f"   Samples: {result['n_samples']}")
        print(f"   Features: {result['n_features']}")
        
        print("\n   Performance Metrics:")
        print("   Training:")
        for metric, value in result['performance']['train'].items():
            print(f"     - {metric}: {value:.3f}")
        
        print("\n   Test:")
        for metric, value in result['performance']['test'].items():
            print(f"     - {metric}: {value:.3f}")
        
        if result.get('feature_importance'):
            print("\n   Feature Importance:")
            for feat, imp in list(result['feature_importance'].items())[:5]:
                print(f"     - {feat}: {imp:.3f}")
        
        # Test 4: Generate model report
        print("\n6. Generating model report...")
        report = server.ml_model_builder.generate_model_report(result['model_id'])
        print("   Report generated successfully")
        print(f"   Report preview: {report[:200]}...")
        
        print("\n✅ ML functionality test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error testing ML functionality: {e}")
        import traceback
        traceback.print_exc()
        
    # Test 5: Test regression functionality
    print("\n7. Testing regression functionality...")
    try:
        from server.tools.regression_analyzer import RegressionAnalyzer
        regression = RegressionAnalyzer()
        
        # Simple linear regression
        reg_result = regression.fit_model(
            cohort_df,
            outcome='mortality',
            predictors=features[:1],  # Use just one predictor
            model_type='logistic'
        )
        
        print("   Regression analysis completed")
        print(f"   Result preview: {reg_result[:200]}...")
        
    except Exception as e:
        print(f"   Warning: Regression test failed: {e}")
    
    # Test 6: Test descriptive stats
    print("\n8. Testing descriptive statistics...")
    try:
        from server.tools.stats_calculator import StatsCalculator
        stats = StatsCalculator()
        
        stats_result = stats.calculate_descriptive_stats(
            cohort_df,
            variables=['admission_year', 'mortality']
        )
        
        print("   Descriptive stats calculated")
        print(f"   Result preview: {stats_result[:200]}...")
        
    except Exception as e:
        print(f"   Warning: Stats test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("- Server initialization: ✅")
    print("- Cohort building: ✅")
    print("- ML model training: ✅" if 'result' in locals() else "- ML model training: ❌")
    print("- Model reporting: ✅" if 'report' in locals() else "- Model reporting: ❌")
    print("- Regression analysis: ✅" if 'reg_result' in locals() else "- Regression analysis: ⚠️")
    print("- Descriptive stats: ✅" if 'stats_result' in locals() else "- Descriptive stats: ⚠️")


if __name__ == "__main__":
    asyncio.run(test_ml_functionality())