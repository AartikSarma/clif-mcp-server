# CLIF Cohort Building Examples

This document provides examples of how to build cohorts using the CLIF MCP Server.

## Basic Age-Based Cohort

To build a cohort of patients within a specific age range:

```json
{
  "criteria": {
    "age_range": {
      "min": 65,
      "max": 85
    }
  }
}
```

## Lab-Based Cohort Examples

### Creatinine > 1.5 in First 24 Hours

To identify patients with elevated creatinine in the first 24 hours of admission:

```json
{
  "criteria": {
    "age_range": {
      "min": 65,
      "max": 85
    },
    "lab_criteria": [
      {
        "lab_name": "Creatinine",
        "condition": "above",
        "value": 1.5,
        "time_window_hours": 24
      }
    ]
  }
}
```

### AKI (Acute Kidney Injury) - 50% Increase in Creatinine

To identify patients with AKI defined as 50% increase in creatinine within 48 hours:

```json
{
  "criteria": {
    "lab_criteria": [
      {
        "lab_name": "Creatinine",
        "condition": "increase",
        "value": 1.5,
        "time_window_hours": 48,
        "baseline_window_hours": 24
      }
    ]
  }
}
```

### Thrombocytopenia (Platelet Count < 100)

```json
{
  "criteria": {
    "lab_criteria": [
      {
        "lab_name": "Platelet Count",
        "condition": "below",
        "value": 100,
        "time_window_hours": 72
      }
    ]
  }
}
```

## Complex Cohorts

### Elderly Patients with Multiple Conditions

```json
{
  "criteria": {
    "age_range": {
      "min": 70,
      "max": 90
    },
    "require_mechanical_ventilation": true,
    "lab_criteria": [
      {
        "lab_name": "Creatinine",
        "condition": "above",
        "value": 2.0,
        "time_window_hours": 48
      }
    ],
    "medications": ["norepinephrine", "vasopressin"]
  }
}
```

## Lab Criteria Options

The `lab_criteria` field supports several conditions:

- **`above`**: Lab value exceeds threshold
- **`below`**: Lab value is below threshold
- **`between`**: Lab value is within a range (requires `min_value` and `max_value`)
- **`increase`**: Lab value increases by a factor or absolute amount
- **`decrease`**: Lab value decreases by a factor or absolute amount

### Time Windows

- `time_window_hours`: Look for lab values within this many hours from admission
- `baseline_window_hours`: For increase/decrease conditions, defines the baseline period

## Natural Language Examples

When using the MCP interface with natural language, these requests map to the criteria above:

1. "Build a cohort of patients between 65-85 with Cr > 1.5 in the first 24 hours"
   → Uses age_range and lab_criteria with condition="above"

2. "Find patients who developed AKI (50% increase in creatinine within 48 hours)"
   → Uses lab_criteria with condition="increase" and value=1.5

3. "Identify ventilated patients over 70 with thrombocytopenia"
   → Combines age_range, require_mechanical_ventilation, and lab_criteria

## Important Notes

1. Lab names must match exactly as they appear in the data (case-insensitive)
2. Time windows are calculated from admission time
3. Multiple lab criteria are combined with AND logic
4. All criteria in the cohort definition must be satisfied