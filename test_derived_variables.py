#!/usr/bin/env python
"""
Test script for derived variable functionality
"""

import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from server.tools.derived_variable_creator import DerivedVariableCreator

def test_derived_variables():
    """Test the derived variable creator"""
    
    # Initialize with synthetic data
    data_path = "./data/synthetic"
    creator = DerivedVariableCreator(data_path)
    
    print("Testing Derived Variable Creator")
    print("=" * 50)
    
    # Test 1: Create mortality variable
    print("\n1. Creating mortality variable (hospitalization level)...")
    mortality_df = creator.create_derived_variable(
        variable_name="mortality",
        variable_type="binary",
        level="hospitalization",
        definition={}
    )
    print(f"   Created {len(mortality_df)} records")
    print(f"   Mortality rate: {mortality_df['mortality'].mean():.2%}")
    
    # Test 2: Create LOS variable
    print("\n2. Creating length of stay variable...")
    los_df = creator.create_derived_variable(
        variable_name="los_days",
        variable_type="numeric",
        level="hospitalization",
        definition={}
    )
    print(f"   Created {len(los_df)} records")
    print(f"   Mean LOS: {los_df['los_days'].mean():.1f} days")
    
    # Test 3: Create ICU LOS variable
    print("\n3. Creating ICU length of stay variable...")
    icu_los_df = creator.create_derived_variable(
        variable_name="icu_los_days",
        variable_type="numeric",
        level="hospitalization",
        definition={}
    )
    print(f"   Created {len(icu_los_df)} records")
    print(f"   Mean ICU LOS: {icu_los_df['icu_los_days'].mean():.1f} days")
    
    # Test 4: Create patient-level variable
    print("\n4. Creating total hospitalizations per patient...")
    total_hosp_df = creator.create_derived_variable(
        variable_name="total_hospitalizations",
        variable_type="numeric",
        level="patient",
        definition={}
    )
    print(f"   Created {len(total_hosp_df)} patient records")
    print(f"   Mean hospitalizations per patient: {total_hosp_df['total_hospitalizations'].mean():.1f}")
    
    # Test 5: Create custom variable with definition
    print("\n5. Creating custom variable: patients with high creatinine...")
    custom_df = creator.create_derived_variable(
        variable_name="high_creatinine",
        variable_type="binary",
        level="hospitalization",
        definition={
            "source_table": "labs",
            "filters": {
                "lab_category": "Creatinine"
            },
            "aggregation": {
                "method": "max",
                "field": "lab_value",
                "group_by": "hospitalization_id"
            },
            "transformation": {
                "type": "binary",
                "threshold": 1.5,
                "operator": ">"
            }
        }
    )
    print(f"   Created {len(custom_df)} records")
    if 'high_creatinine' in custom_df.columns:
        print(f"   High creatinine rate: {custom_df['high_creatinine'].mean():.2%}")
    
    # Test 6: List all predefined variables
    print("\n6. Available predefined variables:")
    all_vars = creator.get_common_derived_variables()
    for level_name, variables in all_vars.items():
        print(f"\n   {level_name}:")
        for var_name, var_info in variables.items():
            print(f"     - {var_name}: {var_info['description']}")
    
    print("\n✓ All tests completed successfully!")

if __name__ == "__main__":
    test_derived_variables()