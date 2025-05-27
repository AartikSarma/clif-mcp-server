"""
Dynamic CLIF Data Dictionary
Dynamically determines available variables based on actual dataset
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime


class DynamicCLIFDictionary:
    """Dynamically builds data dictionary from actual CLIF dataset"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.tables = {}
        self.variable_info = {}
        self.derived_variables = {}
        self._load_data()
        self._build_dictionary()
        
    def _load_data(self):
        """Load all available CLIF tables"""
        csv_files = list(self.data_path.glob("*.csv"))
        for csv_file in csv_files:
            table_name = csv_file.stem
            try:
                # Read with low memory to handle large files
                df = pd.read_csv(csv_file, nrows=1000)  # Sample for schema detection
                self.tables[table_name] = df
            except Exception as e:
                print(f"Warning: Could not load {table_name}: {e}")
                
    def _infer_variable_type(self, series: pd.Series) -> Dict[str, Any]:
        """Infer variable type and metadata from pandas series"""
        info = {}
        
        # Remove nulls for analysis
        non_null = series.dropna()
        
        if len(non_null) == 0:
            info["type"] = "unknown"
            info["all_null"] = True
            return info
            
        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            info["type"] = "datetime"
            info["min"] = non_null.min().isoformat() if len(non_null) > 0 else None
            info["max"] = non_null.max().isoformat() if len(non_null) > 0 else None
        # Try to parse as datetime if string
        elif series.dtype == 'object':
            try:
                # Try common datetime formats silently
                parsed = pd.to_datetime(non_null.iloc[:100], errors='coerce', format='mixed')
                if parsed.notna().sum() > len(parsed) * 0.8:  # 80% parseable as datetime
                    info["type"] = "datetime"
                    info["format_hint"] = "string representation"
                else:
                    # Check if categorical
                    unique_vals = non_null.unique()
                    if len(unique_vals) < 50 and len(unique_vals) < len(non_null) * 0.1:
                        info["type"] = "categorical"
                        info["values"] = sorted([str(v) for v in unique_vals])
                        info["n_categories"] = len(unique_vals)
                    else:
                        info["type"] = "string"
                        info["n_unique"] = len(unique_vals)
            except:
                info["type"] = "string"
        # Numeric types
        elif pd.api.types.is_numeric_dtype(series):
            # Check if likely binary/boolean
            unique_vals = non_null.unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                info["type"] = "binary"
                info["values"] = [0, 1]
            elif pd.api.types.is_integer_dtype(series) and len(unique_vals) < 20:
                info["type"] = "categorical_numeric"
                info["values"] = sorted([int(v) for v in unique_vals])
            else:
                info["type"] = "numeric"
                info["min"] = float(non_null.min())
                info["max"] = float(non_null.max())
                info["mean"] = float(non_null.mean())
                info["median"] = float(non_null.median())
                
        # Add completeness info
        info["completeness"] = len(non_null) / len(series) if len(series) > 0 else 0
        info["n_missing"] = len(series) - len(non_null)
        
        return info
        
    def _build_dictionary(self):
        """Build comprehensive data dictionary from loaded tables"""
        
        # Build dictionary for each table
        for table_name, df in self.tables.items():
            self.variable_info[table_name] = {}
            
            for column in df.columns:
                var_info = self._infer_variable_type(df[column])
                var_info["column_name"] = column
                var_info["table"] = table_name
                
                # Add descriptions based on common CLIF variable patterns
                var_info["description"] = self._get_variable_description(table_name, column)
                
                self.variable_info[table_name][column] = var_info
                
        # Identify derived variables that can be calculated
        self._identify_derived_variables()
        
    def _get_variable_description(self, table: str, column: str) -> str:
        """Get standardized description for common CLIF variables"""
        
        # Common patterns
        descriptions = {
            # IDs
            "patient_id": "Unique patient identifier",
            "hospitalization_id": "Unique identifier for each hospitalization",
            "encounter_id": "Unique encounter identifier",
            
            # Timestamps
            "admission_dttm": "Hospital admission date and time",
            "discharge_dttm": "Hospital discharge date and time",
            "in_dttm": "Unit/location admission time",
            "out_dttm": "Unit/location discharge time",
            "recorded_dttm": "Time when measurement was recorded",
            "admin_dttm": "Time medication was administered",
            "start_dttm": "Start time of intervention/state",
            "stop_dttm": "Stop time of intervention/state",
            "lab_order_dttm": "Time lab was ordered",
            "lab_result_dttm": "Time lab result was available",
            "assessment_dttm": "Time of clinical assessment",
            
            # Demographics
            "age_at_admission": "Patient age in years at time of admission",
            "sex": "Patient biological sex",
            "race": "Patient race",
            "ethnicity": "Patient ethnicity",
            "date_of_birth": "Patient date of birth",
            
            # Clinical
            "admission_source": "Source of hospital admission",
            "discharge_disposition": "Discharge outcome/destination",
            "location_category": "Type of hospital unit/location",
            "vital_category": "Type of vital sign measurement",
            "vital_value": "Numeric value of vital sign",
            "lab_category": "Type of laboratory test",
            "lab_value": "Numeric result of lab test",
            "lab_unit": "Unit of measurement for lab",
            "device_category": "Type of respiratory support device",
            "fio2": "Fraction of inspired oxygen (21-100)",
            "medication_name": "Name of medication",
            "medication_category": "Class of medication",
            "route": "Route of medication administration",
            "dose": "Medication dose amount",
            "dose_unit": "Unit of medication dose",
            "assessment_category": "Type of clinical assessment",
            "assessment_value": "Assessment score/value",
            "position": "Patient position"
        }
        
        # Check exact match first
        if column in descriptions:
            return descriptions[column]
            
        # Check patterns
        if column.endswith("_id"):
            return f"Identifier for {column.replace('_id', '').replace('_', ' ')}"
        elif column.endswith("_dttm") or column.endswith("_time"):
            return f"Timestamp for {column.replace('_dttm', '').replace('_time', '').replace('_', ' ')}"
        elif column.endswith("_value"):
            return f"Numeric value for {column.replace('_value', '').replace('_', ' ')}"
        elif column.endswith("_category") or column.endswith("_type"):
            return f"Category/type of {column.replace('_category', '').replace('_type', '').replace('_', ' ')}"
            
        return f"Variable: {column.replace('_', ' ').title()}"
        
    def _identify_derived_variables(self):
        """Identify which derived variables can be calculated from available data"""
        
        # Check for mortality calculation
        if 'hospitalization' in self.tables:
            hosp_df = self.tables['hospitalization']
            if 'discharge_disposition' in hosp_df.columns:
                self.derived_variables['mortality'] = {
                    "description": "Binary indicator of in-hospital death",
                    "calculation": "discharge_disposition contains 'Expired' or similar",
                    "type": "binary",
                    "requires": ["hospitalization.discharge_disposition"],
                    "available": True
                }
                
            # Check for LOS calculation
            if 'admission_dttm' in hosp_df.columns and 'discharge_dttm' in hosp_df.columns:
                self.derived_variables['los_days'] = {
                    "description": "Hospital length of stay in days",
                    "calculation": "(discharge_dttm - admission_dttm) in days",
                    "type": "numeric",
                    "requires": ["hospitalization.admission_dttm", "hospitalization.discharge_dttm"],
                    "available": True
                }
                
        # Check for ICU LOS
        if 'adt' in self.tables:
            adt_df = self.tables['adt']
            if 'location_category' in adt_df.columns and 'in_dttm' in adt_df.columns and 'out_dttm' in adt_df.columns:
                if 'ICU' in str(adt_df['location_category'].unique()):
                    self.derived_variables['icu_los_days'] = {
                        "description": "Total ICU length of stay in days",
                        "calculation": "Sum of time in ICU locations",
                        "type": "numeric",
                        "requires": ["adt.location_category", "adt.in_dttm", "adt.out_dttm"],
                        "available": True
                    }
                    
        # Check for ventilation
        if 'respiratory_support' in self.tables:
            resp_df = self.tables['respiratory_support']
            if 'device_category' in resp_df.columns:
                self.derived_variables['ventilation'] = {
                    "description": "Binary indicator of mechanical ventilation",
                    "calculation": "Any invasive mechanical ventilation use",
                    "type": "binary",
                    "requires": ["respiratory_support.device_category"],
                    "available": True
                }
                
                if 'start_dttm' in resp_df.columns and 'stop_dttm' in resp_df.columns:
                    self.derived_variables['ventilator_days'] = {
                        "description": "Total days on mechanical ventilation",
                        "calculation": "Sum of invasive ventilation periods",
                        "type": "numeric",
                        "requires": ["respiratory_support.device_category", 
                                    "respiratory_support.start_dttm", 
                                    "respiratory_support.stop_dttm"],
                        "available": True
                    }
                    
        # Check for vasopressor use
        if 'medication_administration' in self.tables:
            med_df = self.tables['medication_administration']
            if 'medication_category' in med_df.columns or 'medication_name' in med_df.columns:
                self.derived_variables['vasopressor_use'] = {
                    "description": "Binary indicator of vasopressor use",
                    "calculation": "Any vasopressor medication administration",
                    "type": "binary",
                    "requires": ["medication_administration.medication_category or medication_name"],
                    "available": True
                }
                
        # Check for lab-based variables
        if 'labs' in self.tables:
            labs_df = self.tables['labs']
            if 'lab_category' in labs_df.columns and 'lab_value' in labs_df.columns:
                lab_categories = labs_df['lab_category'].unique() if 'lab_category' in labs_df.columns else []
                
                if 'Lactate' in str(lab_categories):
                    self.derived_variables['max_lactate'] = {
                        "description": "Maximum lactate value during stay",
                        "calculation": "Maximum lactate lab value",
                        "type": "numeric",
                        "requires": ["labs.lab_category", "labs.lab_value"],
                        "available": True
                    }
                    
                if 'Creatinine' in str(lab_categories):
                    self.derived_variables['max_creatinine'] = {
                        "description": "Maximum creatinine value during stay",
                        "calculation": "Maximum creatinine lab value",
                        "type": "numeric",
                        "requires": ["labs.lab_category", "labs.lab_value"],
                        "available": True
                    }
                    
        # Check for delirium
        if 'patient_assessments' in self.tables:
            assess_df = self.tables['patient_assessments']
            if 'assessment_category' in assess_df.columns:
                categories = assess_df['assessment_category'].unique()
                if 'CAM-ICU' in str(categories):
                    self.derived_variables['delirium'] = {
                        "description": "Binary indicator of positive CAM-ICU",
                        "calculation": "Any positive CAM-ICU assessment",
                        "type": "binary",
                        "requires": ["patient_assessments.assessment_category", 
                                    "patient_assessments.assessment_value"],
                        "available": True
                    }
                    
    def get_variable_info(self, table_name: Optional[str] = None, 
                         variable_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about variables"""
        if table_name and variable_name:
            return self.variable_info.get(table_name, {}).get(variable_name, {})
        elif table_name:
            table_info = self.variable_info.get(table_name, {})
            # Also check if it's a derived variable
            if table_name == "derived_variables":
                return self.derived_variables
            return table_info
        else:
            # Return all tables plus derived variables
            all_info = dict(self.variable_info)
            all_info["derived_variables"] = self.derived_variables
            return all_info
            
    def get_all_variable_names(self) -> List[str]:
        """Get a flat list of all variable names"""
        all_vars = set()
        for table, variables in self.variable_info.items():
            all_vars.update(variables.keys())
        return sorted(list(all_vars))
        
    def get_derived_variable_names(self) -> List[str]:
        """Get list of available derived variable names"""
        return [var for var, info in self.derived_variables.items() 
                if info.get('available', False)]
                
    def get_tables_for_variable(self, variable_name: str) -> List[str]:
        """Find which tables contain a variable"""
        tables = []
        for table, variables in self.variable_info.items():
            if variable_name in variables:
                tables.append(table)
        return tables
        
    def suggest_variables_for_analysis(self, analysis_type: str) -> Dict[str, Any]:
        """Suggest relevant variables based on what's actually available"""
        
        available_vars = self.get_all_variable_names()
        available_derived = self.get_derived_variable_names()
        
        suggestions = {
            "mortality_prediction": {
                "outcome": "mortality" if "mortality" in available_derived else None,
                "predictors": [],
                "note": ""
            },
            "los_prediction": {
                "outcome": "los_days" if "los_days" in available_derived else None,
                "predictors": [],
                "note": ""
            },
            "delirium_risk": {
                "outcome": "delirium" if "delirium" in available_derived else None,
                "predictors": [],
                "note": ""
            }
        }
        
        # Build predictor lists based on available variables
        if analysis_type == "mortality_prediction":
            potential_predictors = [
                "age_at_admission", "sex", "race", "ventilation",
                "vasopressor_use", "max_lactate", "max_creatinine"
            ]
            suggestions["mortality_prediction"]["predictors"] = [
                p for p in potential_predictors 
                if p in available_vars or p in available_derived
            ]
            if not suggestions["mortality_prediction"]["outcome"]:
                suggestions["mortality_prediction"]["note"] = "Mortality outcome not available - need discharge_disposition in hospitalization table"
                
        elif analysis_type == "los_prediction":
            potential_predictors = [
                "age_at_admission", "sex", "admission_source",
                "ventilation", "delirium", "max_creatinine"
            ]
            suggestions["los_prediction"]["predictors"] = [
                p for p in potential_predictors
                if p in available_vars or p in available_derived
            ]
            if "icu_los_days" in available_derived:
                suggestions["los_prediction"]["note"] = "ICU LOS also available for ICU-specific analyses"
                
        elif analysis_type == "delirium_risk":
            potential_predictors = [
                "age_at_admission", "sex", "ventilation"
            ]
            suggestions["delirium_risk"]["predictors"] = [
                p for p in potential_predictors
                if p in available_vars or p in available_derived
            ]
            if not suggestions["delirium_risk"]["outcome"]:
                suggestions["delirium_risk"]["note"] = "Delirium outcome not available - need CAM-ICU assessments"
                
        return suggestions.get(analysis_type, {})


# Global instance that will be initialized when needed
_dictionary_instance: Optional[DynamicCLIFDictionary] = None


def initialize_dictionary(data_path: str):
    """Initialize the global dictionary instance"""
    global _dictionary_instance
    _dictionary_instance = DynamicCLIFDictionary(data_path)
    return _dictionary_instance


def get_dictionary() -> Optional[DynamicCLIFDictionary]:
    """Get the current dictionary instance"""
    return _dictionary_instance


# Legacy interface for backwards compatibility
def get_variable_info(table_name=None, variable_name=None):
    """Get information about CLIF variables"""
    if _dictionary_instance:
        return _dictionary_instance.get_variable_info(table_name, variable_name)
    return {}


def get_all_variable_names():
    """Get a flat list of all variable names"""
    if _dictionary_instance:
        return _dictionary_instance.get_all_variable_names()
    return []


def get_derived_variable_names():
    """Get list of derived variable names"""
    if _dictionary_instance:
        return _dictionary_instance.get_derived_variable_names()
    return []


def suggest_variables_for_analysis(analysis_type):
    """Suggest relevant variables for different analysis types"""
    if _dictionary_instance:
        return _dictionary_instance.suggest_variables_for_analysis(analysis_type)
    return {}


def get_variable_info(table_name=None, variable_name=None):
    """Get information about CLIF variables"""
    if table_name and variable_name:
        return CLIF_DATA_DICTIONARY.get(table_name, {}).get(variable_name, {})
    elif table_name:
        return CLIF_DATA_DICTIONARY.get(table_name, {})
    else:
        return CLIF_DATA_DICTIONARY


def get_all_variable_names():
    """Get a flat list of all variable names"""
    all_vars = []
    for table, variables in CLIF_DATA_DICTIONARY.items():
        if table != "derived_variables":
            all_vars.extend(variables.keys())
    return sorted(list(set(all_vars)))


def get_derived_variable_names():
    """Get list of derived variable names"""
    return list(CLIF_DATA_DICTIONARY.get("derived_variables", {}).keys())


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