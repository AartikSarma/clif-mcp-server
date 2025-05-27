#!/usr/bin/env python
"""
Demonstration of CLIF MCP Server Analyses
This script shows the types of analyses the MCP server performs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.tools.data_explorer import DataExplorer
from server.tools.cohort_builder import CohortBuilder
from server.tools.outcomes_analyzer import OutcomesAnalyzer
from server.tools.code_generator import CodeGenerator
import pandas as pd

def main():
    print("CLIF MCP Server Analysis Demonstration")
    print("=" * 60)
    
    data_path = "./data/synthetic"
    
    # 1. Explore the data schema
    print("\n1. EXPLORING DATA SCHEMA")
    print("-" * 40)
    explorer = DataExplorer(data_path)
    schema_info = explorer.explore_schema()
    print(schema_info[:500] + "...")
    
    # 2. Build a cohort
    print("\n\n2. BUILDING PATIENT COHORT")
    print("-" * 40)
    print("Criteria: Ventilated patients aged 65+ with ICU stay > 3 days")
    
    cohort_builder = CohortBuilder(data_path)
    criteria = {
        'age_range': {'min': 65, 'max': 120},
        'icu_los_range': {'min': 3, 'max': float('inf')},
        'require_mechanical_ventilation': True,
        'exclude_readmissions': False
    }
    
    cohort, cohort_id = cohort_builder.build_cohort(criteria)
    print(f"Cohort ID: {cohort_id}")
    print(f"Cohort size: {len(cohort)} hospitalizations")
    print(f"Unique patients: {cohort['patient_id'].nunique()}")
    
    # Get cohort characteristics
    characteristics = cohort_builder.get_cohort_characteristics(cohort)
    print(f"Mean age: {characteristics['demographics']['age_mean']:.1f} years")
    print(f"Mortality rate: {characteristics['clinical']['mortality_rate']:.1%}")
    print(f"Mean ICU LOS: {characteristics['clinical']['icu_los_mean']:.1f} days")
    
    # 3. Analyze outcomes
    print("\n\n3. ANALYZING OUTCOMES")
    print("-" * 40)
    
    outcomes_analyzer = OutcomesAnalyzer(data_path)
    outcomes = ['mortality', 'icu_los', 'ventilator_days', 'aki', 'delirium']
    
    results = outcomes_analyzer.analyze_outcomes(
        cohort_id=None,  # Use the cohort we just built
        outcomes=outcomes,
        stratify_by=[],
        adjust_for=[]
    )
    
    # Since we're not using saved cohorts, analyze our cohort directly
    # This simulates what the MCP server would do internally
    print("Mortality Analysis:")
    mortality = (cohort['discharge_disposition'] == 'Expired').mean()
    print(f"  Rate: {mortality:.1%}")
    
    print("\nICU Length of Stay:")
    print(f"  Median: {characteristics['clinical']['icu_los_median']:.1f} days")
    print(f"  Mean: {characteristics['clinical']['icu_los_mean']:.1f} days")
    
    # 4. Compare two cohorts (early vs late ventilation)
    print("\n\n4. COMPARING COHORTS")
    print("-" * 40)
    print("Comparing early vs late mechanical ventilation")
    
    # Build comparison cohorts
    # Early ventilation: ventilated within first 2 days
    early_vent_criteria = {
        'age_range': {'min': 18, 'max': 120},
        'require_mechanical_ventilation': True,
        'icu_los_range': {'min': 0, 'max': 2}
    }
    
    late_vent_criteria = {
        'age_range': {'min': 18, 'max': 120},
        'require_mechanical_ventilation': True,
        'icu_los_range': {'min': 3, 'max': float('inf')}
    }
    
    early_cohort, early_id = cohort_builder.build_cohort(early_vent_criteria)
    late_cohort, late_id = cohort_builder.build_cohort(late_vent_criteria)
    
    print(f"Early ventilation cohort: {len(early_cohort)} patients")
    print(f"Late ventilation cohort: {len(late_cohort)} patients")
    
    # Compare mortality
    early_mortality = (early_cohort['discharge_disposition'] == 'Expired').mean()
    late_mortality = (late_cohort['discharge_disposition'] == 'Expired').mean()
    
    print(f"\nMortality comparison:")
    print(f"  Early ventilation: {early_mortality:.1%}")
    print(f"  Late ventilation: {late_mortality:.1%}")
    print(f"  Risk difference: {(late_mortality - early_mortality):.1%}")
    
    # 5. Generate analysis code
    print("\n\n5. GENERATING ANALYSIS CODE")
    print("-" * 40)
    
    code_generator = CodeGenerator()
    
    # Store our analysis
    code_generator.store_analysis({
        'cohort_criteria': criteria,
        'outcomes': {
            'mortality': {'rate': mortality, 'count': int(mortality * len(cohort))},
            'icu_los': {'median': characteristics['clinical']['icu_los_median']}
        }
    })
    
    # Generate Python code
    python_code = code_generator.generate_code(
        analysis_type='outcomes_analysis',
        language='python',
        include_visualizations=True
    )
    
    # Save the generated code
    with open('./generated_analysis.py', 'w') as f:
        f.write(python_code)
    
    print("Generated Python analysis script: ./generated_analysis.py")
    print("\nFirst 20 lines of generated code:")
    print("-" * 40)
    print('\n'.join(python_code.split('\n')[:20]))
    
    # 6. Privacy demonstration
    print("\n\n6. PRIVACY PROTECTION DEMONSTRATION")
    print("-" * 40)
    
    from server.security.privacy_guard import PrivacyGuard
    privacy_guard = PrivacyGuard()
    
    # Example with small cell suppression
    unsafe_output = {
        'cohort_size': 3,  # Too small!
        'mortality_count': 2,
        'patient_id': 'PT000123'  # Should be removed
    }
    
    safe_output = privacy_guard.sanitize_output(unsafe_output)
    print(f"Unsafe output: {unsafe_output}")
    print(f"Sanitized output: {safe_output}")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("\nThis script shows what the MCP server does when you:")
    print("1. Explore the CLIF schema")
    print("2. Build cohorts with clinical criteria")
    print("3. Analyze outcomes (mortality, LOS, complications)")
    print("4. Compare cohorts statistically")
    print("5. Generate reproducible analysis code")
    print("6. Protect patient privacy")


if __name__ == "__main__":
    main()