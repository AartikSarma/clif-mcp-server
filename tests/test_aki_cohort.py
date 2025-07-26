#!/usr/bin/env python
"""
Test AKI cohort building functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from server.tools.cohort_builder import CohortBuilder

# Initialize cohort builder
cohort_builder = CohortBuilder("./data/synthetic")

print("Testing AKI Cohort Building")
print("=" * 50)

# Example 1: Build AKI cohort using KDIGO criteria
print("\nExample 1: AKI Cohort (KDIGO Criteria)")
print("-" * 30)

# User defines their own AKI criteria
aki_criteria = {
    'lab_criteria': [
        {
            'lab_name': 'Creatinine',
            'condition': 'increase',
            'value': 1.5,  # 1.5x increase
            'time_window_hours': 48,
            'baseline_window_hours': 168  # 7 days baseline
        }
    ]
}

try:
    cohort, cohort_id = cohort_builder.build_cohort(aki_criteria)
    print(f"AKI cohort built successfully!")
    print(f"Cohort ID: {cohort_id}")
    print(f"Size: {len(cohort)} hospitalizations")
    
    # Get characteristics
    chars = cohort_builder.get_cohort_characteristics(cohort)
    print(f"Mortality rate: {chars['clinical']['mortality_rate']:.1%}")
    print(f"Mean age: {chars['demographics']['age_mean']:.1f} years")
except Exception as e:
    print(f"Error building AKI cohort: {e}")

# Example 2: Build cohort with AKI AND elevated lactate
print("\n\nExample 2: AKI + Sepsis (Elevated Lactate)")
print("-" * 30)

aki_sepsis_criteria = {
    'lab_criteria': [
        # AKI criteria
        {
            'lab_name': 'Creatinine',
            'condition': 'increase',
            'value': 1.5,
            'time_window_hours': 48,
            'baseline_window_hours': 168
        },
        # Sepsis criteria
        {
            'lab_name': 'Lactate',
            'condition': 'above',
            'value': 2.0,
            'time_window_hours': 24
        }
    ]
}

try:
    cohort, cohort_id = cohort_builder.build_cohort(aki_sepsis_criteria)
    print(f"AKI + Sepsis cohort built successfully!")
    print(f"Cohort ID: {cohort_id}")
    print(f"Size: {len(cohort)} hospitalizations")
except Exception as e:
    print(f"Error: {e}")

# Example 3: Custom lab criteria
print("\n\nExample 3: Custom Lab Criteria")
print("-" * 30)

custom_criteria = {
    'age_range': {'min': 18, 'max': 80},
    'lab_criteria': [
        {
            'lab_name': 'Hemoglobin',
            'condition': 'below',
            'value': 7.0  # Severe anemia
        }
    ],
    'require_mechanical_ventilation': True
}

try:
    cohort, cohort_id = cohort_builder.build_cohort(custom_criteria)
    print(f"Custom cohort built successfully!")
    print(f"Cohort ID: {cohort_id}")
    print(f"Size: {len(cohort)} hospitalizations")
    print("Criteria: Age 18-80, Hemoglobin < 7, on mechanical ventilation")
except Exception as e:
    print(f"Error: {e}")

print("\n\nHow to use in Claude Desktop:")
print("-" * 50)
print("""
To build an AKI cohort, use the build_cohort tool with:

{
  "criteria": {
    "lab_criteria": [
      {
        "lab_name": "Creatinine",
        "condition": "increase",
        "value": 1.5,
        "time_window_hours": 48,
        "baseline_window_hours": 168
      }
    ]
  }
}

Or for absolute increase:

{
  "criteria": {
    "lab_criteria": [
      {
        "lab_name": "Creatinine", 
        "condition": "increase",
        "absolute_increase": 0.3,
        "time_window_hours": 48
      }
    ]
  }
}
""")