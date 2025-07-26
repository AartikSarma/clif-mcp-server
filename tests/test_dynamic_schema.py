#!/usr/bin/env python
"""
Test script for dynamic schema loading functionality
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from server.tools.schema_loader import SchemaLoader
from server.tools.cohort_builder_dynamic import DynamicCohortBuilder
from server.tools.data_explorer_dynamic import DynamicDataExplorer


def test_schema_loader():
    """Test basic schema loading functionality"""
    print("Testing Schema Loader")
    print("=" * 50)
    
    # Test with default CLIF schema
    schema_path = "schema_config.json"
    data_path = "./data/synthetic"
    
    loader = SchemaLoader(schema_path, data_path)
    
    # Discover files
    print("\n1. Discovering files...")
    discovered = loader.discover_files()
    for table, files in discovered.items():
        print(f"  {table}: {len(files)} file(s)")
    
    # Get schema info
    print("\n2. Schema Summary:")
    print(loader.get_schema_summary())
    
    # Test loading a table
    print("\n3. Loading patient table...")
    patient_df = loader.load_table('patient')
    if patient_df is not None:
        print(f"  Loaded {len(patient_df)} patients")
        print(f"  Columns: {list(patient_df.columns)[:5]}...")
    
    # Validate relationships
    print("\n4. Validating relationships...")
    validation_results = loader.validate_relationships()
    if validation_results:
        for table, issues in validation_results.items():
            print(f"  {table}:")
            for issue in issues:
                print(f"    - {issue}")
    else:
        print("  All relationships valid!")


def test_dynamic_cohort_builder():
    """Test cohort building with dynamic schema"""
    print("\n\nTesting Dynamic Cohort Builder")
    print("=" * 50)
    
    schema_path = "schema_config.json"
    data_path = "./data/synthetic"
    
    builder = DynamicCohortBuilder(schema_path, data_path)
    
    # Build a simple cohort
    criteria = {
        'age_range': {'min': 18, 'max': 65},
        'require_mechanical_ventilation': True
    }
    
    print("\n1. Building cohort with criteria:")
    print(json.dumps(criteria, indent=2))
    
    try:
        cohort, cohort_id = builder.build_cohort(criteria)
        print(f"\n2. Built cohort: {cohort_id}")
        print(f"   Size: {len(cohort)} patients")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test with custom filters
    criteria_custom = {
        'custom_filters': [
            {
                'table': 'labs',
                'column': 'test_name',
                'operator': 'contains',
                'value': 'creatinine'
            }
        ]
    }
    
    print("\n3. Building cohort with custom filters:")
    print(json.dumps(criteria_custom, indent=2))
    
    try:
        cohort2, cohort_id2 = builder.build_cohort(criteria_custom)
        print(f"   Built cohort: {cohort_id2}")
        print(f"   Size: {len(cohort2)} patients")
    except Exception as e:
        print(f"   Error: {e}")


def test_dynamic_data_explorer():
    """Test data exploration with dynamic schema"""
    print("\n\nTesting Dynamic Data Explorer")
    print("=" * 50)
    
    schema_path = "schema_config.json"
    data_path = "./data/synthetic"
    
    explorer = DynamicDataExplorer(schema_path, data_path)
    
    # Explore all tables
    print("\n1. Exploring all tables:")
    print(explorer.explore_schema())
    
    # Explore specific table
    print("\n2. Exploring 'patient' table in detail:")
    print(explorer.explore_schema('patient'))
    
    # Validate schema
    print("\n3. Validating schema:")
    validation_report = explorer.validate_schema()
    print(f"   Overall status: {validation_report['overall_status']}")
    for table, info in validation_report['tables'].items():
        if info['issues']:
            print(f"   {table}: {info['status']} - {len(info['issues'])} issues")


def test_custom_schema():
    """Test with a custom schema configuration"""
    print("\n\nTesting Custom Schema")
    print("=" * 50)
    
    # This would work if you had data files matching the custom schema
    custom_schema_path = "examples/custom_schema_example.json"
    
    if Path(custom_schema_path).exists():
        print(f"\n1. Loading custom schema from {custom_schema_path}")
        
        # Just load and display the schema
        with open(custom_schema_path, 'r') as f:
            custom_schema = json.load(f)
        
        print(f"   Name: {custom_schema['name']}")
        print(f"   Version: {custom_schema['version']}")
        print(f"   Tables: {list(custom_schema['tables'].keys())}")
        
        # You could test with custom data by creating matching files
        # For now, just show how it would work
        print("\n2. To use this schema:")
        print("   - Create data files matching the file patterns")
        print("   - Initialize tools with the custom schema path")
        print("   - All tools will automatically adapt to the new schema")
    

if __name__ == "__main__":
    # Run tests
    test_schema_loader()
    test_dynamic_cohort_builder()
    test_dynamic_data_explorer()
    test_custom_schema()
    
    print("\n\nDynamic schema testing complete!")