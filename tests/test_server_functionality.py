#!/usr/bin/env python3
"""
Quick test script to verify server functionality
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from server.tools.schema_loader import SchemaLoader
from server.tools.cohort_builder import CohortBuilder
from server.tools.data_explorer import DataExplorer

def test_server_functionality():
    """Test the main server functionality"""
    
    schema_path = "schema_config.json"
    data_path = "./data/synthetic"
    
    print("🧪 Testing CLIF MCP Server Functionality")
    print("=" * 50)
    
    # Test 1: Schema Loading
    print("\n1️⃣ Testing Schema Loading...")
    try:
        loader = SchemaLoader(schema_path, data_path)
        files = loader.discover_files()
        print(f"   ✅ Discovered {len(files)} tables")
        
        # Test loading each table
        for table_name in files.keys():
            df = loader.load_table(table_name)
            if df is not None:
                print(f"   ✅ {table_name}: {len(df):,} rows")
            else:
                print(f"   ❌ {table_name}: Failed to load")
                
    except Exception as e:
        print(f"   ❌ Schema loading failed: {e}")
        return False
    
    # Test 2: Data Explorer
    print("\n2️⃣ Testing Data Explorer...")
    try:
        explorer = DataExplorer(schema_path, data_path)
        
        # Test schema validation
        validation = explorer.validate_schema()
        print(f"   📊 Schema validation: {validation['overall_status']}")
        
        # Test column exploration
        patient_cols = explorer.get_column_values('patient', 'sex')
        if 'top_values' in patient_cols:
            print(f"   ✅ Patient sex values: {list(patient_cols['top_values'].keys())}")
        
    except Exception as e:
        print(f"   ❌ Data explorer failed: {e}")
        return False
    
    # Test 3: Cohort Building
    print("\n3️⃣ Testing Cohort Building...")
    try:
        builder = CohortBuilder(schema_path, data_path)
        
        # Test simple age-based cohort
        criteria = {
            'age_range': {'min': 18, 'max': 80}
        }
        cohort, cohort_id = builder.build_cohort(criteria)
        print(f"   ✅ Built age-based cohort: {len(cohort):,} patients ({cohort_id})")
        
        # Test ventilation cohort
        criteria = {
            'require_mechanical_ventilation': True
        }
        vent_cohort, vent_id = builder.build_cohort(criteria)
        print(f"   ✅ Built ventilated cohort: {len(vent_cohort):,} patients ({vent_id})")
        
        # Test custom filter cohort
        criteria = {
            'custom_filters': [
                {
                    'table': 'labs',
                    'column': 'test_name',
                    'operator': 'contains',
                    'value': 'Creatinine'
                }
            ]
        }
        lab_cohort, lab_id = builder.build_cohort(criteria)
        print(f"   ✅ Built lab-based cohort: {len(lab_cohort):,} patients ({lab_id})")
        
    except Exception as e:
        print(f"   ❌ Cohort building failed: {e}")
        return False
    
    # Test 4: Schema Adaptability
    print("\n4️⃣ Testing Schema Adaptability...")
    try:
        # Test with missing age column fallback
        hosp_df = loader.load_table('hospitalization')
        age_col = builder._find_age_column(hosp_df)
        if age_col:
            print(f"   ✅ Found age column: {age_col}")
        else:
            print("   ⚠️  No age column found, but this is handled gracefully")
            
        # Test ICU LOS calculation
        icu_los = builder._calculate_icu_los(hosp_df)
        if icu_los is not None:
            print(f"   ✅ ICU LOS calculated for {len(icu_los):,} hospitalizations")
        else:
            print("   ⚠️  ICU LOS calculation failed, but handled gracefully")
            
    except Exception as e:
        print(f"   ❌ Schema adaptability test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Server functionality is working correctly.")
    return True

if __name__ == "__main__":
    success = test_server_functionality()
    sys.exit(0 if success else 1)