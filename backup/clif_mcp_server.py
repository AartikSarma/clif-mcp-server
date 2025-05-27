#!/usr/bin/env python
"""
CLIF MCP Server - Enhanced with automatic data dictionary
"""

import asyncio
import json
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from server.tools.data_explorer import DataExplorer
from server.tools.cohort_builder import CohortBuilder
from server.tools.outcomes_analyzer import OutcomesAnalyzer
from server.tools.code_generator import CodeGenerator
from server.tools.ml_model_builder import MLModelBuilder
from server.tools.comprehensive_dictionary import ComprehensiveDictionary

# Load CLIF data dictionary from JSON file
def load_data_dictionary():
    """Load the CLIF data dictionary from JSON file"""
    dict_path = Path(__file__).parent / "clif_data_dictionary.json"
    with open(dict_path, 'r') as f:
        return json.load(f)

# Load dictionary on import
CLIF_DATA_DICTIONARY = load_data_dictionary()

def get_all_variable_names():
    """Get a flat list of all variable names"""
    all_vars = []
    for table, variables in CLIF_DATA_DICTIONARY.items():
        if table != "derived_variables":
            all_vars.extend(variables.keys())
    return sorted(list(set(all_vars)))

def suggest_variables_for_analysis(analysis_type):
    """Suggest relevant variables for different analysis types"""
    suggestions = {
        "mortality_prediction": {
            "outcome": "mortality",
            "predictors": ["age_at_admission", "sex", "race", "ventilation", 
                          "vasopressor_use", "max_lactate", "max_creatinine"],
            "note": "Consider adding SOFA score components if available"
        },
        "los_prediction": {
            "outcome": "los_days",
            "predictors": ["age_at_admission", "sex", "admission_source", 
                          "ventilation", "delirium", "max_creatinine"],
            "note": "ICU LOS might be more specific for ICU-only analyses"
        },
        "delirium_risk": {
            "outcome": "delirium",
            "predictors": ["age_at_admission", "sex", "ventilation", 
                          "medication_category:Sedative", "medication_category:Analgesic"],
            "note": "Requires CAM-ICU assessments in data"
        },
        "aki_risk": {
            "outcome": "aki",  # Would need to be derived
            "predictors": ["age_at_admission", "baseline_creatinine", "max_creatinine",
                          "vasopressor_use", "nephrotoxic_medications"],
            "note": "AKI definition requires baseline and peak creatinine"
        }
    }
    return suggestions.get(analysis_type, {})

# Initialize server
app = Server("clif-mcp-server")

# Global variables for tools
data_explorer = None
cohort_builder = None
outcomes_analyzer = None
code_generator = None
current_cohort = None
data_dictionary = {}


def build_data_dictionary():
    """Build a comprehensive data dictionary from available tables"""
    global data_dictionary
    
    try:
        # Use the comprehensive dictionary
        data_dictionary = CLIF_DATA_DICTIONARY.copy()
        
        # Add dynamic information from actual data
        data_path = Path(data_explorer.data_path)
        
        # Get actual categorical values from data
        categorical_values = {}
        
        # Check each table for categorical variables
        for table_name in data_explorer.tables:
            try:
                df = pd.read_csv(data_path / f"{table_name}.csv", nrows=1000)
                table_dict = CLIF_DATA_DICTIONARY.get(table_name, {})
                
                for col_name, col_info in table_dict.items():
                    if col_info.get("type") == "categorical" and col_name in df.columns:
                        unique_vals = df[col_name].dropna().unique()
                        categorical_values[f"{table_name}.{col_name}"] = list(unique_vals)[:20]  # Limit to 20 values
            except Exception as e:
                print(f"Could not read {table_name}: {e}", file=sys.stderr)
        
        data_dictionary["actual_categorical_values"] = categorical_values
        data_dictionary["available_tables"] = data_explorer.tables
        
        # Add analysis suggestions
        data_dictionary["analysis_suggestions"] = {
            "mortality_prediction": suggest_variables_for_analysis("mortality_prediction"),
            "los_prediction": suggest_variables_for_analysis("los_prediction"),
            "delirium_risk": suggest_variables_for_analysis("delirium_risk")
        }
        
    except Exception as e:
        print(f"Error building data dictionary: {e}", file=sys.stderr)
        data_dictionary = {"error": "Could not build data dictionary"}


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available CLIF analysis tools"""
    
    # Build data dictionary if not already built
    if not data_dictionary and data_explorer:
        build_data_dictionary()
    
    # Create a comprehensive description including variable names
    variable_info = ""
    if data_dictionary:
        variable_info = """

AVAILABLE VARIABLES:
Core Variables:
- age_at_admission: Patient age in years
- sex: Patient sex (requires merging with patient table)
- race: Patient race (requires merging with patient table)
- admission_source: How patient was admitted
- discharge_disposition: Discharge outcome
- mortality: Binary death outcome (derived)
- los_days: Hospital length of stay (derived)
- ventilation: Mechanical ventilation status (derived)

Use exact variable names as listed above."""
    
    return [
        Tool(
            name="show_data_dictionary",
            description="Show comprehensive data dictionary with all available variables and their descriptions",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="explore_data",
            description="Explore CLIF dataset schema and summary statistics" + variable_info,
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Specific table to explore (optional)"
                    }
                }
            }
        ),
        Tool(
            name="build_cohort",
            description="Build a patient cohort with clinical criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "age_min": {"type": "number", "description": "Minimum age"},
                    "age_max": {"type": "number", "description": "Maximum age"},
                    "require_ventilation": {"type": "boolean", "description": "Require mechanical ventilation"},
                    "icu_los_min": {"type": "number", "description": "Minimum ICU length of stay in days"},
                    "exclude_readmissions": {"type": "boolean", "description": "Exclude readmissions"}
                },
                "required": []
            }
        ),
        Tool(
            name="fit_regression",
            description=f"""Fit regression models with various outcomes and predictors.
            
COMMON VARIABLES:
Outcomes: mortality, los_days, ventilator_days
Predictors: age_at_admission, sex, race, ventilation, admission_source
Model types: linear, logistic, poisson, negative_binomial, cox, mixed_linear

Example: outcome='mortality', predictors=['age_at_admission', 'ventilation'], model_type='logistic'""",
            inputSchema={
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "description": "Outcome variable (e.g., 'mortality', 'los_days', 'icu_los_days', 'ventilator_days')"
                    },
                    "predictors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Predictor variables (e.g., ['age_at_admission', 'sex', 'admission_source'])"
                    },
                    "model_type": {
                        "type": "string",
                        "enum": ["linear", "logistic", "poisson", "negative_binomial", "cox", "mixed_linear", "mixed_logistic"],
                        "description": "Type of regression model"
                    },
                    "random_effects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Random effects for mixed models (e.g., ['patient_id'])"
                    },
                    "interaction_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Interaction terms (e.g., ['age_at_admission*sex'])"
                    },
                    "offset": {
                        "type": "string",
                        "description": "Offset variable for count models"
                    }
                },
                "required": ["outcome", "predictors", "model_type"]
            }
        ),
        Tool(
            name="descriptive_stats",
            description=f"""Generate descriptive statistics and tables.
            
AVAILABLE VARIABLES: {', '.join(['age_at_admission', 'sex', 'race', 'mortality', 'los_days', 'ventilation', 'admission_source', 'discharge_disposition'])}""",
            inputSchema={
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to summarize"
                    },
                    "stratify_by": {
                        "type": "string",
                        "description": "Variable to stratify by (optional)"
                    },
                    "include_tests": {
                        "type": "boolean",
                        "description": "Include statistical tests for group comparisons"
                    }
                },
                "required": ["variables"]
            }
        ),
        Tool(
            name="generate_code",
            description="Generate reproducible analysis code",
            inputSchema={
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "r"],
                        "description": "Programming language"
                    },
                    "include_regression": {
                        "type": "boolean",
                        "description": "Include last regression model in generated code"
                    }
                },
                "required": ["language"]
            }
        ),
        Tool(
            name="build_ml_model",
            description="Build flexible machine learning models for clinical prediction",
            inputSchema={
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "description": "Target variable to predict (e.g., mortality, los_days)"
                    },
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of feature columns to use for prediction"
                    },
                    "model_type": {
                        "type": "string",
                        "description": "Model type (auto, logistic_regression, random_forest, gradient_boosting, etc.)",
                        "default": "auto"
                    },
                    "feature_selection": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "enum": ["kbest", "mutual_info", "rfe"]
                            },
                            "n_features": {"type": "integer"}
                        },
                        "description": "Optional feature selection configuration"
                    },
                    "validation_split": {
                        "type": "number",
                        "default": 0.2,
                        "description": "Test set proportion (0-1)"
                    }
                },
                "required": ["outcome", "features"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute CLIF analysis tools"""
    global current_cohort
    
    try:
        if name == "show_data_dictionary":
            if not data_dictionary:
                build_data_dictionary()
            
            result = """COMPREHENSIVE CLIF DATA DICTIONARY
=====================================

"""
            # Show each table's variables
            for table_name in ['hospitalization', 'patient', 'adt', 'vitals', 'labs', 
                             'respiratory_support', 'medication_administration', 
                             'patient_assessments', 'position']:
                if table_name in data_dictionary:
                    result += f"\n{table_name.upper()} TABLE:\n"
                    table_vars = data_dictionary[table_name]
                    for var_name, var_info in table_vars.items():
                        desc = var_info.get('description', '')
                        var_type = var_info.get('type', '')
                        result += f"  • {var_name} ({var_type}): {desc}\n"
                        
                        # Show categorical values if available
                        if var_type == 'categorical' and 'values' in var_info:
                            result += f"    Values: {', '.join(var_info['values'][:5])}"
                            if len(var_info['values']) > 5:
                                result += "..."
                            result += "\n"
            
            # Show derived variables
            result += "\nDERIVED VARIABLES (calculated from raw data):\n"
            for var_name, var_info in data_dictionary.get('derived_variables', {}).items():
                desc = var_info.get('description', '')
                calc = var_info.get('calculation', '')
                result += f"  • {var_name}: {desc}\n"
                result += f"    Calculation: {calc}\n"
            
            # Show common analysis patterns
            result += "\nCOMMON ANALYSIS PATTERNS:\n"
            
            result += "\n1. MORTALITY PREDICTION:\n"
            mort_suggest = data_dictionary.get('analysis_suggestions', {}).get('mortality_prediction', {})
            if mort_suggest:
                result += f"   Outcome: {mort_suggest.get('outcome')}\n"
                result += f"   Predictors: {', '.join(mort_suggest.get('predictors', []))}\n"
            
            result += "\n2. LENGTH OF STAY PREDICTION:\n"
            los_suggest = data_dictionary.get('analysis_suggestions', {}).get('los_prediction', {})
            if los_suggest:
                result += f"   Outcome: {los_suggest.get('outcome')}\n"
                result += f"   Predictors: {', '.join(los_suggest.get('predictors', []))}\n"
            
            result += "\n3. TIME-SERIES VARIABLES (from vitals/labs):\n"
            result += "   • Heart Rate, Blood Pressure, Temperature, SpO2\n"
            result += "   • Hemoglobin, WBC, Creatinine, Lactate\n"
            result += "   Note: These require aggregation (mean, max, min, trend)\n"
            
            result += "\n4. MEDICATION EXPOSURES:\n"
            result += "   • Vasopressors: Norepinephrine, Epinephrine, Vasopressin\n"
            result += "   • Sedatives: Propofol, Midazolam, Dexmedetomidine\n"
            result += "   • Antibiotics: Various classes available\n"
            
            result += "\nTIP: Use exact variable names as shown above in your analyses!"
            
            return [TextContent(type="text", text=result)]
        
        elif name == "explore_data":
            table_name = arguments.get("table_name")
            result = data_explorer.explore_schema(table_name)
            
            # Add variable name reminder
            if not table_name:
                result += "\n\nTIP: Use 'show_data_dictionary' tool to see all available variable names and their exact spelling."
            
            return [TextContent(type="text", text=result)]
            
        elif name == "build_cohort":
            # Build criteria from arguments
            criteria = {}
            
            if "age_min" in arguments or "age_max" in arguments:
                criteria["age_range"] = {
                    "min": arguments.get("age_min", 0),
                    "max": arguments.get("age_max", 120)
                }
            
            if arguments.get("require_ventilation"):
                criteria["require_mechanical_ventilation"] = True
                
            if "icu_los_min" in arguments:
                criteria["icu_los_range"] = {
                    "min": arguments.get("icu_los_min", 0),
                    "max": float('inf')
                }
                
            if arguments.get("exclude_readmissions"):
                criteria["exclude_readmissions"] = True
            
            # Build cohort
            cohort, cohort_id = cohort_builder.build_cohort(criteria)
            current_cohort = cohort
            
            # Get characteristics
            chars = cohort_builder.get_cohort_characteristics(cohort)
            
            result = f"""Cohort built successfully!
Cohort ID: {cohort_id}
Size: {len(cohort)} hospitalizations
Unique patients: {cohort['patient_id'].nunique()}

Demographics:
- Mean age: {chars['demographics']['age_mean']:.1f} years
- Mortality rate: {chars['clinical']['mortality_rate']:.1%}
- ICU LOS (median): {chars['clinical']['icu_los_median']:.1f} days
- Ventilation rate: {chars['clinical']['ventilation_rate']:.1%}

Available variables for analysis:
- From hospitalization: {', '.join([col for col in cohort.columns if col not in ['patient_id', 'hospitalization_id']])}
- Additional (requires merge): sex, race, ethnicity
- Derived: mortality, los_days, ventilation, icu_los_days"""
            
            return [TextContent(type="text", text=result)]
            
        elif name == "fit_regression":
            if current_cohort is None:
                return [TextContent(type="text", text="Please build a cohort first using build_cohort")]
            
            outcome = arguments["outcome"]
            predictors = arguments["predictors"]
            model_type = arguments["model_type"]
            random_effects = arguments.get("random_effects", [])
            interaction_terms = arguments.get("interaction_terms", [])
            offset = arguments.get("offset")
            
            # Validate variable names
            valid_outcomes = ['mortality', 'los_days', 'icu_los_days', 'ventilator_days'] + list(current_cohort.columns)
            if outcome not in valid_outcomes:
                return [TextContent(type="text", text=f"Invalid outcome '{outcome}'. Valid options: {', '.join(valid_outcomes[:10])}...\nUse 'show_data_dictionary' for complete list.")]
            
            # Prepare the data
            analysis_df = current_cohort.copy()
            
            # Create outcome variables
            if outcome == "mortality":
                analysis_df['mortality'] = (analysis_df['discharge_disposition'] == 'Expired').astype(int)
            elif outcome == "los_days":
                analysis_df['los_days'] = (pd.to_datetime(analysis_df['discharge_dttm']) - 
                                          pd.to_datetime(analysis_df['admission_dttm'])).dt.total_seconds() / 86400
            elif outcome == "icu_los_days":
                # Calculate ICU LOS from ADT data
                adt_df = pd.read_csv(Path(data_explorer.data_path) / "adt.csv")
                icu_los = []
                for hosp_id in analysis_df['hospitalization_id']:
                    icu_stays = adt_df[(adt_df['hospitalization_id'] == hosp_id) & 
                                      (adt_df['location_category'] == 'ICU')]
                    if len(icu_stays) > 0:
                        total_hours = 0
                        for _, stay in icu_stays.iterrows():
                            duration = (pd.to_datetime(stay['out_dttm']) - 
                                      pd.to_datetime(stay['in_dttm'])).total_seconds() / 3600
                            total_hours += duration
                        icu_los.append(total_hours / 24)
                    else:
                        icu_los.append(0)
                analysis_df['icu_los_days'] = icu_los
            
            # Prepare predictors - merge with patient data if needed
            if any(var in ['sex', 'race', 'ethnicity'] for var in predictors):
                patient_df = pd.read_csv(Path(data_explorer.data_path) / "patient.csv")
                analysis_df = analysis_df.merge(patient_df[['patient_id', 'sex', 'race', 'ethnicity']], 
                                              on='patient_id', how='left')
            
            # Check for ventilation status if needed
            if 'ventilation' in predictors:
                resp_df = pd.read_csv(Path(data_explorer.data_path) / "respiratory_support.csv")
                vent_ids = resp_df[resp_df['device_category'] == 'Invasive Mechanical Ventilation']['hospitalization_id'].unique()
                analysis_df['ventilation'] = analysis_df['hospitalization_id'].isin(vent_ids).astype(int)
            
            # Add interaction terms
            for term in interaction_terms:
                if '*' in term:
                    var1, var2 = term.split('*')
                    if var1 in analysis_df.columns and var2 in analysis_df.columns:
                        # Handle numeric * categorical interactions
                        if analysis_df[var1].dtype in ['float64', 'int64'] and analysis_df[var2].dtype == 'object':
                            # Create dummy variables for categorical
                            dummies = pd.get_dummies(analysis_df[var2], prefix=var2)
                            for col in dummies.columns:
                                analysis_df[f"{var1}*{col}"] = analysis_df[var1] * dummies[col]
                        else:
                            analysis_df[term] = analysis_df[var1] * analysis_df[var2]
                        predictors.append(term)
            
            # Check which predictors are missing
            missing_predictors = [p for p in predictors if p not in analysis_df.columns and '*' not in p]
            if missing_predictors:
                return [TextContent(type="text", text=f"Missing predictors: {', '.join(missing_predictors)}\nAvailable: {', '.join(analysis_df.columns)}\nUse 'show_data_dictionary' for help.")]
            
            # Handle missing data
            predictor_cols = [p for p in predictors if p in analysis_df.columns]
            analysis_df = analysis_df.dropna(subset=[outcome] + predictor_cols)
            
            # Fit the model
            try:
                if model_type == "linear":
                    from sklearn.linear_model import LinearRegression
                    from scipy import stats
                    
                    X = pd.get_dummies(analysis_df[predictor_cols], drop_first=True)
                    y = analysis_df[outcome]
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Calculate statistics
                    predictions = model.predict(X)
                    residuals = y - predictions
                    n = len(y)
                    p = X.shape[1]
                    
                    # R-squared
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    r_squared = 1 - (ss_res/ss_tot)
                    adj_r_squared = 1 - (1-r_squared)*(n-1)/(n-p-1)
                    
                    # Standard errors and p-values
                    mse = ss_res / (n - p - 1)
                    var_coef = mse * np.linalg.inv(X.T @ X).diagonal()
                    se_coef = np.sqrt(var_coef)
                    t_stats = model.coef_ / se_coef
                    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
                    
                    result = f"""Linear Regression Results:
Outcome: {outcome}
N = {n} observations
Predictors: {', '.join(predictor_cols)}

Model Performance:
- R-squared: {r_squared:.3f}
- Adjusted R-squared: {adj_r_squared:.3f}
- RMSE: {np.sqrt(mse):.3f}

Coefficients:
- Intercept: {model.intercept_:.3f}
"""
                    for i, col in enumerate(X.columns):
                        result += f"- {col}: {model.coef_[i]:.3f} (SE: {se_coef[i]:.3f}, p={p_values[i]:.3f})\n"
                    
                elif model_type == "logistic":
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.metrics import roc_auc_score, confusion_matrix
                    
                    X = pd.get_dummies(analysis_df[predictor_cols], drop_first=True)
                    y = analysis_df[outcome]
                    
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X, y)
                    
                    # Calculate AUC
                    y_pred_proba = model.predict_proba(X)[:, 1]
                    auc = roc_auc_score(y, y_pred_proba)
                    
                    # Confusion matrix
                    y_pred = model.predict(X)
                    cm = confusion_matrix(y, y_pred)
                    sensitivity = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
                    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
                    
                    # Odds ratios
                    odds_ratios = np.exp(model.coef_[0])
                    
                    result = f"""Logistic Regression Results:
Outcome: {outcome}
N = {len(y)} observations
Events: {y.sum()} ({y.mean():.1%})
Predictors: {', '.join(predictor_cols)}

Model Performance:
- AUC: {auc:.3f}
- Sensitivity: {sensitivity:.3f}
- Specificity: {specificity:.3f}

Odds Ratios:
- Intercept: {np.exp(model.intercept_[0]):.3f}
"""
                    for i, col in enumerate(X.columns):
                        result += f"- {col}: {odds_ratios[i]:.3f}\n"
                
                elif model_type == "poisson":
                    import statsmodels.api as sm
                    
                    X = pd.get_dummies(analysis_df[predictor_cols], drop_first=True)
                    X = sm.add_constant(X)
                    y = analysis_df[outcome]
                    
                    if offset:
                        model = sm.GLM(y, X, family=sm.families.Poisson(), 
                                     offset=analysis_df[offset]).fit()
                    else:
                        model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
                    
                    result = f"""Poisson Regression Results:
Outcome: {outcome}
N = {len(y)} observations
Mean count: {y.mean():.2f}
Predictors: {', '.join(predictor_cols)}

Deviance: {model.deviance:.2f}
Pearson chi2: {model.pearson_chi2:.2f}

Coefficients (Rate Ratios):
"""
                    for i, coef_name in enumerate(model.params.index):
                        rr = np.exp(model.params[i])
                        ci_low = np.exp(model.conf_int()[0][i])
                        ci_high = np.exp(model.conf_int()[1][i])
                        result += f"- {coef_name}: {rr:.3f} (95% CI: {ci_low:.3f}-{ci_high:.3f}, p={model.pvalues[i]:.3f})\n"
                
                elif model_type == "negative_binomial":
                    import statsmodels.api as sm
                    
                    X = pd.get_dummies(analysis_df[predictor_cols], drop_first=True)
                    X = sm.add_constant(X)
                    y = analysis_df[outcome]
                    
                    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
                    
                    result = f"""Negative Binomial Regression Results:
Outcome: {outcome}
N = {len(y)} observations
Mean count: {y.mean():.2f}
Variance: {y.var():.2f}
Predictors: {', '.join(predictor_cols)}

Deviance: {model.deviance:.2f}

Coefficients (Rate Ratios):
"""
                    for i, coef_name in enumerate(model.params.index):
                        rr = np.exp(model.params[i])
                        ci_low = np.exp(model.conf_int()[0][i])
                        ci_high = np.exp(model.conf_int()[1][i])
                        result += f"- {coef_name}: {rr:.3f} (95% CI: {ci_low:.3f}-{ci_high:.3f}, p={model.pvalues[i]:.3f})\n"
                
                elif model_type == "mixed_linear":
                    import statsmodels.formula.api as smf
                    
                    # Create formula
                    formula = f"{outcome} ~ " + " + ".join(predictor_cols)
                    
                    # Add random effects
                    if random_effects:
                        re_formula = " + ".join([f"(1|{re})" for re in random_effects])
                        
                        # Use R-style mixed model syntax
                        result = f"""Mixed Linear Model Results:
Outcome: {outcome}
Fixed effects: {', '.join(predictor_cols)}
Random effects: {', '.join(random_effects)}

Note: For mixed effects models, please use the generated R code or install pymer4.
The generated code will include the proper lme4 syntax for R."""
                    
                elif model_type == "cox":
                    from lifelines import CoxPHFitter
                    
                    # For Cox model, need time and event variables
                    analysis_df['time'] = analysis_df['los_days']
                    analysis_df['event'] = (analysis_df['discharge_disposition'] == 'Expired').astype(int)
                    
                    cph = CoxPHFitter()
                    cph.fit(analysis_df[['time', 'event'] + predictor_cols], 
                           duration_col='time', event_col='event')
                    
                    result = f"""Cox Proportional Hazards Results:
N = {len(analysis_df)} observations
Events: {analysis_df['event'].sum()}
Predictors: {', '.join(predictor_cols)}

Concordance: {cph.concordance_index_:.3f}

Hazard Ratios:
"""
                    summary = cph.summary
                    for var in summary.index:
                        hr = summary.loc[var, 'exp(coef)']
                        ci_low = summary.loc[var, 'exp(coef) lower 95%']
                        ci_high = summary.loc[var, 'exp(coef) upper 95%']
                        p_val = summary.loc[var, 'p']
                        result += f"- {var}: {hr:.3f} (95% CI: {ci_low:.3f}-{ci_high:.3f}, p={p_val:.3f})\n"
                
                else:
                    result = f"Model type '{model_type}' not yet implemented. Currently supported: linear, logistic, poisson, negative_binomial, cox, mixed_linear"
                
                # Store regression config for code generation
                app.last_regression_config = {
                    'outcome': outcome,
                    'predictors': predictor_cols,
                    'model_type': model_type,
                    'random_effects': random_effects,
                    'interaction_terms': interaction_terms
                }
                
            except Exception as e:
                result = f"""Error fitting {model_type} model: {str(e)}

Common issues:
1. Variable name typos - use 'show_data_dictionary' to see exact names
2. Wrong outcome type for model (e.g., continuous outcome for logistic regression)
3. Missing data - some variables may require merging additional tables

Available variables in current cohort: {', '.join(analysis_df.columns[:15])}..."""
            
            return [TextContent(type="text", text=result)]
        
        elif name == "descriptive_stats":
            if current_cohort is None:
                return [TextContent(type="text", text="Please build a cohort first using build_cohort")]
            
            variables = arguments["variables"]
            stratify_by = arguments.get("stratify_by")
            include_tests = arguments.get("include_tests", False)
            
            analysis_df = current_cohort.copy()
            
            # Add derived variables if requested
            if 'mortality' in variables:
                analysis_df['mortality'] = (analysis_df['discharge_disposition'] == 'Expired').astype(int)
            if 'los_days' in variables:
                analysis_df['los_days'] = (pd.to_datetime(analysis_df['discharge_dttm']) - 
                                          pd.to_datetime(analysis_df['admission_dttm'])).dt.total_seconds() / 86400
            
            # Merge additional data if needed
            if any(var in ['sex', 'race', 'ethnicity'] for var in variables + ([stratify_by] if stratify_by else [])):
                patient_df = pd.read_csv(Path(data_explorer.data_path) / "patient.csv")
                analysis_df = analysis_df.merge(patient_df[['patient_id', 'sex', 'race', 'ethnicity']], 
                                              on='patient_id', how='left')
            
            # Check for missing variables
            missing_vars = [v for v in variables if v not in analysis_df.columns]
            if missing_vars:
                available = list(analysis_df.columns)
                return [TextContent(type="text", text=f"Missing variables: {', '.join(missing_vars)}\nAvailable: {', '.join(available[:20])}...\nUse 'show_data_dictionary' for help.")]
            
            result = f"Descriptive Statistics\nN = {len(analysis_df)} observations\n\n"
            
            if stratify_by and stratify_by in analysis_df.columns:
                groups = analysis_df[stratify_by].unique()
                result += f"Stratified by: {stratify_by}\n"
                result += "-" * 50 + "\n\n"
                
                for group in groups:
                    group_df = analysis_df[analysis_df[stratify_by] == group]
                    result += f"\n{stratify_by} = {group} (n={len(group_df)}):\n"
                    
                    for var in variables:
                        if var in group_df.columns:
                            if group_df[var].dtype in ['float64', 'int64']:
                                result += f"  {var}: {group_df[var].mean():.2f} ± {group_df[var].std():.2f}\n"
                            else:
                                counts = group_df[var].value_counts()
                                result += f"  {var}:\n"
                                for val, count in counts.head(5).items():
                                    result += f"    {val}: {count} ({count/len(group_df)*100:.1f}%)\n"
            else:
                for var in variables:
                    if var in analysis_df.columns:
                        if analysis_df[var].dtype in ['float64', 'int64']:
                            result += f"\n{var}:\n"
                            result += f"  Mean: {analysis_df[var].mean():.2f}\n"
                            result += f"  SD: {analysis_df[var].std():.2f}\n"
                            result += f"  Median: {analysis_df[var].median():.2f}\n"
                            result += f"  IQR: {analysis_df[var].quantile(0.25):.2f} - {analysis_df[var].quantile(0.75):.2f}\n"
                        else:
                            result += f"\n{var}:\n"
                            counts = analysis_df[var].value_counts()
                            for val, count in counts.head(10).items():
                                result += f"  {val}: {count} ({count/len(analysis_df)*100:.1f}%)\n"
            
            return [TextContent(type="text", text=result)]
            
        elif name == "generate_code":
            language = arguments["language"]
            include_regression = arguments.get("include_regression", False)
            
            # Check if we have stored regression results
            has_regression = hasattr(app, 'last_regression_config') and include_regression
            
            if language == "python":
                code = '''#!/usr/bin/env python
"""
CLIF Outcomes Analysis
Generated by CLIF MCP Server

Variable Reference:
- age_at_admission: Patient age in years
- sex: Patient sex (Male/Female) - requires patient table merge
- race: Patient race - requires patient table merge  
- mortality: Binary death outcome (derived from discharge_disposition=='Expired')
- los_days: Hospital length of stay in days
- ventilation: Mechanical ventilation status (from respiratory_support table)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH = Path("./data")  # Update to your CLIF data path

# Load data
print("Loading CLIF data...")
hospitalization = pd.read_csv(DATA_PATH / "hospitalization.csv")
patient = pd.read_csv(DATA_PATH / "patient.csv")
adt = pd.read_csv(DATA_PATH / "adt.csv")
respiratory_support = pd.read_csv(DATA_PATH / "respiratory_support.csv")

# Parse dates
date_cols = ['admission_dttm', 'discharge_dttm']
for col in date_cols:
    hospitalization[col] = pd.to_datetime(hospitalization[col])

# Build cohort (update criteria as needed)
cohort = hospitalization[
    (hospitalization['age_at_admission'] >= 18) & 
    (hospitalization['age_at_admission'] <= 120)
]

# Merge with patient data
cohort = cohort.merge(patient, on='patient_id', how='left')

print(f"Cohort size: {len(cohort)}")

# Create outcome variables
cohort['mortality'] = (cohort['discharge_disposition'] == 'Expired').astype(int)
cohort['los_days'] = (cohort['discharge_dttm'] - cohort['admission_dttm']).dt.total_seconds() / 86400

# Create predictor variables
# Ventilation status
vent_ids = respiratory_support[
    respiratory_support['device_category'] == 'Invasive Mechanical Ventilation'
]['hospitalization_id'].unique()
cohort['ventilation'] = cohort['hospitalization_id'].isin(vent_ids).astype(int)

# Basic descriptive statistics
print(f"\\nDescriptive Statistics:")
print(f"Mortality rate: {cohort['mortality'].mean():.1%}")
print(f"Mean LOS: {cohort['los_days'].mean():.1f} days")
print(f"Median LOS: {cohort['los_days'].median():.1f} days")
print(f"Ventilation rate: {cohort['ventilation'].mean():.1%}")

# Variable summary
print(f"\\nAvailable variables:")
print(f"Continuous: age_at_admission, los_days")
print(f"Binary: mortality, ventilation")
print(f"Categorical: sex, race, admission_source, discharge_disposition")
'''

                if has_regression:
                    config = app.last_regression_config
                    
                    if config['model_type'] == 'linear':
                        code += f'''
# Regression Analysis: {config['model_type']}
outcome = '{config['outcome']}'
predictors = {config['predictors']}

# Prepare data
X = pd.get_dummies(cohort[predictors], drop_first=True)
y = cohort[outcome]

# Remove missing
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

# Fit model
model = LinearRegression()
model.fit(X, y)

# Results
print(f"\\nLinear Regression Results:")
print(f"R-squared: {{1 - np.sum((y - model.predict(X))**2) / np.sum((y - y.mean())**2):.3f}}")
print(f"\\nCoefficients:")
print(f"  Intercept: {{model.intercept_:.3f}}")
for i, col in enumerate(X.columns):
    print(f"  {{col}}: {{model.coef_[i]:.3f}}")
'''
                    elif config['model_type'] == 'logistic':
                        code += f'''
# Regression Analysis: {config['model_type']}
outcome = '{config['outcome']}'
predictors = {config['predictors']}

# Prepare data
X = pd.get_dummies(cohort[predictors], drop_first=True)
y = cohort[outcome]

# Remove missing
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

# Fit model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Calculate AUC
y_pred_proba = model.predict_proba(X)[:, 1]
auc = roc_auc_score(y, y_pred_proba)

# Results
print(f"\\nLogistic Regression Results:")
print(f"AUC: {{auc:.3f}}")
print(f"\\nOdds Ratios:")
print(f"  Intercept: {{np.exp(model.intercept_[0]):.3f}}")
for i, col in enumerate(X.columns):
    print(f"  {{col}}: {{np.exp(model.coef_[0][i]):.3f}}")
'''

            else:  # R
                code = '''#!/usr/bin/env Rscript
# CLIF Outcomes Analysis
# Generated by CLIF MCP Server
#
# Variable Reference:
# - age_at_admission: Patient age in years
# - sex: Patient sex (Male/Female) - requires patient table merge
# - race: Patient race - requires patient table merge  
# - mortality: Binary death outcome (derived from discharge_disposition=='Expired')
# - los_days: Hospital length of stay in days
# - ventilation: Mechanical ventilation status (from respiratory_support table)

library(tidyverse)
library(lubridate)
library(lme4)
library(survival)
library(broom)

# Configuration
DATA_PATH <- "./data"  # Update to your CLIF data path

# Load data
cat("Loading CLIF data...\\n")
hospitalization <- read_csv(file.path(DATA_PATH, "hospitalization.csv"))
patient <- read_csv(file.path(DATA_PATH, "patient.csv"))
adt <- read_csv(file.path(DATA_PATH, "adt.csv"))
respiratory_support <- read_csv(file.path(DATA_PATH, "respiratory_support.csv"))

# Build cohort
cohort <- hospitalization %>%
  filter(age_at_admission >= 18 & age_at_admission <= 120) %>%
  left_join(patient, by = "patient_id")

cat(sprintf("Cohort size: %d\\n", nrow(cohort)))

# Create outcome variables
cohort <- cohort %>%
  mutate(
    mortality = as.integer(discharge_disposition == "Expired"),
    los_days = as.numeric(difftime(discharge_dttm, admission_dttm, units = "days"))
  )

# Create predictor variables
vent_ids <- respiratory_support %>%
  filter(device_category == "Invasive Mechanical Ventilation") %>%
  pull(hospitalization_id) %>%
  unique()

cohort <- cohort %>%
  mutate(ventilation = as.integer(hospitalization_id %in% vent_ids))

# Descriptive statistics
cat("\\nDescriptive Statistics:\\n")
cat(sprintf("Mortality rate: %.1f%%\\n", mean(cohort$mortality) * 100))
cat(sprintf("Mean LOS: %.1f days\\n", mean(cohort$los_days)))
cat(sprintf("Median LOS: %.1f days\\n", median(cohort$los_days)))
cat(sprintf("Ventilation rate: %.1f%%\\n", mean(cohort$ventilation) * 100))

# Variable summary
cat("\\nAvailable variables:\\n")
cat("Continuous: age_at_admission, los_days\\n")
cat("Binary: mortality, ventilation\\n") 
cat("Categorical: sex, race, admission_source, discharge_disposition\\n")
'''

                if has_regression:
                    config = app.last_regression_config
                    
                    if config['model_type'] == 'linear':
                        code += f'''
# Regression Analysis: Linear Model
model <- lm({config['outcome']} ~ {' + '.join(config['predictors'])}, data = cohort)
summary(model)
'''
                    elif config['model_type'] == 'logistic':
                        code += f'''
# Regression Analysis: Logistic Model
model <- glm({config['outcome']} ~ {' + '.join(config['predictors'])}, 
             data = cohort, family = binomial())
summary(model)

# Odds ratios with confidence intervals
exp(cbind(OR = coef(model), confint(model)))
'''
                    elif config['model_type'] == 'mixed_linear' and config.get('random_effects'):
                        code += f'''
# Mixed Effects Model
model <- lmer({config['outcome']} ~ {' + '.join(config['predictors'])} + 
              {' + '.join([f'(1|{re})' for re in config['random_effects']])}, 
              data = cohort)
summary(model)
'''
            
            # Save code to output directory
            output_dir = Path.home() / "Desktop"  # Save to user's Desktop
            output_dir.mkdir(exist_ok=True)
            
            filename = f"clif_analysis_{language}.{language[0]}"
            filepath = output_dir / filename
            
            try:
                with open(filepath, 'w') as f:
                    f.write(code)
                
                result = f"Generated {language} analysis code saved to: {filepath}\n\nPreview:\n{code[:500]}..."
            except Exception as e:
                # If can't write to disk, just return the code
                result = f"Could not save file ({e}). Here's the complete {language} code:\n\n{code}"
            
            return [TextContent(type="text", text=result)]
            
        elif name == "build_ml_model":
            outcome = arguments["outcome"]
            features = arguments["features"]
            model_type = arguments.get("model_type", "auto")
            feature_selection = arguments.get("feature_selection")
            validation_split = arguments.get("validation_split", 0.2)
            
            # Load data
            if current_cohort is not None:
                # Use current cohort
                df = current_cohort
            else:
                # Load default data
                patient_df = pd.read_csv(Path(data_explorer.data_path) / "patient.csv")
                hosp_df = pd.read_csv(Path(data_explorer.data_path) / "hospitalization.csv")
                df = patient_df.merge(hosp_df, on="patient_id")
                
                # Add derived outcomes if needed
                if outcome == "mortality" and "mortality" not in df.columns:
                    df["mortality"] = (df["discharge_disposition"] == "Expired").astype(int)
                if outcome == "los_days" and "los_days" not in df.columns:
                    df["admission_dttm"] = pd.to_datetime(df["admission_dttm"])
                    df["discharge_dttm"] = pd.to_datetime(df["discharge_dttm"])
                    df["los_days"] = (df["discharge_dttm"] - df["admission_dttm"]).dt.total_seconds() / 86400
            
            # Build model
            result = ml_model_builder.build_model(
                df=df,
                outcome=outcome,
                features=features,
                model_type=model_type,
                feature_selection=feature_selection,
                validation_split=validation_split,
                hyperparameter_tuning=False  # Disable for speed in MCP
            )
            
            # Format results
            output = f"""ML Model Built Successfully!
Model ID: {result['model_id']}
Model Type: {result['model_type']}
Task: {result['task']}
Samples: {result['n_samples']}
Features: {result['n_features']}

Performance Metrics:
Training:
"""
            for metric, value in result['performance']['train'].items():
                output += f"  - {metric}: {value:.3f}\n"
            
            output += "\nTest:"
            for metric, value in result['performance']['test'].items():
                output += f"  - {metric}: {value:.3f}\n"
            
            if result.get('feature_importance'):
                output += "\nTop Feature Importance:"
                for i, (feat, imp) in enumerate(list(result['feature_importance'].items())[:10]):
                    output += f"\n  {i+1}. {feat}: {imp:.3f}"
            
            return [TextContent(type="text", text=output)]
            
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        error_msg = f"Error executing {name}: {str(e)}"
        print(error_msg, file=sys.stderr)
        return [TextContent(type="text", text=error_msg)]


async def main():
    """Main entry point"""
    global data_explorer, cohort_builder, outcomes_analyzer, code_generator, ml_model_builder, comprehensive_dictionary
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="CLIF MCP Server")
    parser.add_argument("--data-path", default="./data/synthetic", help="Path to CLIF CSV files")
    args = parser.parse_args()
    
    # Initialize tools
    print(f"Initializing CLIF tools with data path: {args.data_path}", file=sys.stderr)
    
    try:
        data_explorer = DataExplorer(args.data_path)
        cohort_builder = CohortBuilder(args.data_path)
        outcomes_analyzer = OutcomesAnalyzer(args.data_path)
        code_generator = CodeGenerator()
        ml_model_builder = MLModelBuilder(args.data_path)
        comprehensive_dictionary = ComprehensiveDictionary(args.data_path)
        
        # Build data dictionary on startup
        build_data_dictionary()
        
        print("CLIF tools initialized successfully", file=sys.stderr)
        print(f"Available variables loaded: {len(data_dictionary.get('hospitalization_variables', {}))} hospitalization, "
              f"{len(data_dictionary.get('patient_variables', {}))} patient variables", file=sys.stderr)
        
    except Exception as e:
        print(f"Error initializing tools: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())