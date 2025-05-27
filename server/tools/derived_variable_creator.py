"""
Derived Variable Creator Tool for CLIF MCP Server
Creates new variables from existing data at different levels (timepoint, hospitalization, patient)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import json


class DerivedVariableCreator:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.tables = {}
        self._load_tables()
        
    def _load_tables(self):
        """Load CLIF tables as needed"""
        table_names = ['patient', 'hospitalization', 'adt', 'vitals', 'labs', 
                      'respiratory_support', 'medication_administration', 
                      'patient_assessments', 'position']
        
        for table in table_names:
            filepath = self.data_path / f"{table}.csv"
            if filepath.exists():
                self.tables[table] = pd.read_csv(filepath)
                # Parse datetime columns
                for col in self.tables[table].columns:
                    if 'dttm' in col or 'date' in col:
                        self.tables[table][col] = pd.to_datetime(
                            self.tables[table][col], errors='coerce'
                        )
    
    def create_derived_variable(self, 
                              variable_name: str,
                              variable_type: str,
                              level: str,
                              definition: Dict[str, Any],
                              save_to_table: Optional[str] = None) -> pd.DataFrame:
        """
        Create a new derived variable based on the definition
        
        Args:
            variable_name: Name of the new variable
            variable_type: Type of variable (binary, categorical, numeric, datetime)
            level: Level at which to create variable (timepoint, hospitalization, patient)
            definition: Dictionary defining how to create the variable
            save_to_table: Optional - which table to save the variable to
            
        Returns:
            DataFrame with the new variable
        """
        
        if level == "hospitalization":
            return self._create_hospitalization_variable(
                variable_name, variable_type, definition
            )
        elif level == "patient":
            return self._create_patient_variable(
                variable_name, variable_type, definition
            )
        elif level == "timepoint":
            return self._create_timepoint_variable(
                variable_name, variable_type, definition
            )
        else:
            raise ValueError(f"Unknown level: {level}")
    
    def _create_hospitalization_variable(self, 
                                       variable_name: str,
                                       variable_type: str,
                                       definition: Dict[str, Any]) -> pd.DataFrame:
        """Create variables at hospitalization level"""
        
        # Start with hospitalization table
        result = self.tables['hospitalization'].copy()
        
        # Common derived variables
        if variable_name == "mortality":
            result['mortality'] = (
                result['discharge_disposition'] == 'Expired'
            ).astype(int)
            
        elif variable_name == "los_days":
            result['los_days'] = (
                result['discharge_dttm'] - result['admission_dttm']
            ).dt.total_seconds() / 86400
            
        elif variable_name == "icu_los_days":
            # Calculate from ADT data
            adt = self.tables['adt']
            icu_stays = adt[adt['location_category'] == 'ICU'].copy()
            icu_los = icu_stays.groupby('hospitalization_id').apply(
                lambda x: ((x['out_dttm'] - x['in_dttm']).sum()).total_seconds() / 86400
            ).reset_index(name='icu_los_days')
            result = result.merge(icu_los, on='hospitalization_id', how='left')
            result['icu_los_days'] = result['icu_los_days'].fillna(0)
            
        elif variable_name == "mechanical_ventilation":
            # Check respiratory support
            resp = self.tables['respiratory_support']
            vent_devices = ['Conventional mechanical ventilation', 
                           'High frequency oscillatory ventilation']
            vent_hosps = resp[resp['device_category'].isin(vent_devices)][
                'hospitalization_id'
            ].unique()
            result['mechanical_ventilation'] = result['hospitalization_id'].isin(
                vent_hosps
            ).astype(int)
            
        elif variable_name == "ventilator_days":
            # Calculate ventilator duration
            resp = self.tables['respiratory_support']
            vent_devices = ['Conventional mechanical ventilation', 
                           'High frequency oscillatory ventilation']
            vent_data = resp[resp['device_category'].isin(vent_devices)].copy()
            vent_duration = vent_data.groupby('hospitalization_id').apply(
                lambda x: ((x['recorded_stop_dttm'] - x['recorded_start_dttm']).sum()).total_seconds() / 86400
            ).reset_index(name='ventilator_days')
            result = result.merge(vent_duration, on='hospitalization_id', how='left')
            result['ventilator_days'] = result['ventilator_days'].fillna(0)
            
        elif variable_name == "max_sofa_score":
            # Maximum SOFA score during hospitalization
            assessments = self.tables['patient_assessments']
            sofa = assessments[assessments['assessment_category'] == 'SOFA']
            max_sofa = sofa.groupby('hospitalization_id')['assessment_value'].max().reset_index()
            max_sofa.columns = ['hospitalization_id', 'max_sofa_score']
            result = result.merge(max_sofa, on='hospitalization_id', how='left')
            
        elif variable_name == "aki_stage":
            # AKI staging based on creatinine criteria
            labs = self.tables['labs']
            creat = labs[labs['lab_category'] == 'Creatinine'].copy()
            
            # Get baseline and peak creatinine for each hospitalization
            creat_stats = creat.groupby('hospitalization_id')['lab_value'].agg([
                'min', 'max', 'first'
            ]).reset_index()
            creat_stats.columns = ['hospitalization_id', 'min_creat', 'max_creat', 'baseline_creat']
            
            # Calculate AKI stage
            creat_stats['creat_ratio'] = creat_stats['max_creat'] / creat_stats['baseline_creat']
            creat_stats['aki_stage'] = 0
            creat_stats.loc[creat_stats['creat_ratio'] >= 1.5, 'aki_stage'] = 1
            creat_stats.loc[creat_stats['creat_ratio'] >= 2.0, 'aki_stage'] = 2
            creat_stats.loc[creat_stats['creat_ratio'] >= 3.0, 'aki_stage'] = 3
            
            result = result.merge(
                creat_stats[['hospitalization_id', 'aki_stage']], 
                on='hospitalization_id', 
                how='left'
            )
            result['aki_stage'] = result['aki_stage'].fillna(0).astype(int)
            
        # Custom definition handling
        elif 'source_table' in definition:
            result = self._apply_custom_definition(
                result, variable_name, variable_type, definition, 'hospitalization'
            )
            
        return result[['hospitalization_id', variable_name]]
    
    def _create_patient_variable(self, 
                               variable_name: str,
                               variable_type: str,
                               definition: Dict[str, Any]) -> pd.DataFrame:
        """Create variables at patient level"""
        
        # Start with patient table
        result = self.tables['patient'].copy()
        hosp = self.tables['hospitalization']
        
        # Common derived variables
        if variable_name == "total_hospitalizations":
            hosp_counts = hosp.groupby('patient_id').size().reset_index(name='total_hospitalizations')
            result = result.merge(hosp_counts, on='patient_id', how='left')
            result['total_hospitalizations'] = result['total_hospitalizations'].fillna(0).astype(int)
            
        elif variable_name == "total_icu_days":
            # Sum of all ICU days across hospitalizations
            adt = self.tables['adt']
            icu_stays = adt[adt['location_category'] == 'ICU'].copy()
            
            # Join with hospitalization to get patient_id
            icu_with_patient = icu_stays.merge(
                hosp[['hospitalization_id', 'patient_id']], 
                on='hospitalization_id'
            )
            
            # Calculate total ICU days per patient
            icu_with_patient['icu_hours'] = (
                icu_with_patient['out_dttm'] - icu_with_patient['in_dttm']
            ).dt.total_seconds() / 3600
            
            total_icu = icu_with_patient.groupby('patient_id')['icu_hours'].sum() / 24
            total_icu = total_icu.reset_index(name='total_icu_days')
            
            result = result.merge(total_icu, on='patient_id', how='left')
            result['total_icu_days'] = result['total_icu_days'].fillna(0)
            
        elif variable_name == "ever_ventilated":
            # Check if patient was ever on mechanical ventilation
            resp = self.tables['respiratory_support']
            vent_devices = ['Conventional mechanical ventilation', 
                           'High frequency oscillatory ventilation']
            
            # Get hospitalizations with ventilation
            vent_hosps = resp[resp['device_category'].isin(vent_devices)]['hospitalization_id']
            
            # Get patients with those hospitalizations
            vent_patients = hosp[hosp['hospitalization_id'].isin(vent_hosps)]['patient_id'].unique()
            
            result['ever_ventilated'] = result['patient_id'].isin(vent_patients).astype(int)
            
        elif variable_name == "mortality_any_admission":
            # Check if patient died in any admission
            mortality_patients = hosp[hosp['discharge_disposition'] == 'Expired']['patient_id'].unique()
            result['mortality_any_admission'] = result['patient_id'].isin(mortality_patients).astype(int)
            
        # Custom definition handling
        elif 'source_table' in definition:
            result = self._apply_custom_definition(
                result, variable_name, variable_type, definition, 'patient'
            )
            
        return result[['patient_id', variable_name]]
    
    def _create_timepoint_variable(self, 
                                 variable_name: str,
                                 variable_type: str,
                                 definition: Dict[str, Any]) -> pd.DataFrame:
        """Create variables at timepoint level (for time-series data)"""
        
        source_table = definition.get('source_table', 'vitals')
        result = self.tables[source_table].copy()
        
        # Examples of timepoint-level derived variables
        if variable_name == "map" and source_table == "vitals":
            # Mean Arterial Pressure from systolic and diastolic
            sbp = result[result['vital_category'] == 'Systolic blood pressure'][
                ['hospitalization_id', 'recorded_dttm', 'vital_value']
            ].rename(columns={'vital_value': 'sbp'})
            
            dbp = result[result['vital_category'] == 'Diastolic blood pressure'][
                ['hospitalization_id', 'recorded_dttm', 'vital_value']
            ].rename(columns={'vital_value': 'dbp'})
            
            # Merge on hospitalization and approximate time
            bp_merged = pd.merge_asof(
                sbp.sort_values('recorded_dttm'),
                dbp.sort_values('recorded_dttm'),
                on='recorded_dttm',
                by='hospitalization_id',
                tolerance=pd.Timedelta(minutes=5)
            )
            
            # Calculate MAP
            bp_merged['map'] = (bp_merged['sbp'] + 2 * bp_merged['dbp']) / 3
            
            # Create result
            result = bp_merged[['hospitalization_id', 'recorded_dttm', 'map']].dropna()
            result['vital_category'] = 'Mean arterial pressure'
            result['vital_value'] = result['map']
            result = result.drop('map', axis=1)
            
        elif variable_name == "shock_index" and source_table == "vitals":
            # Shock Index = Heart Rate / Systolic BP
            hr = result[result['vital_category'] == 'Heart rate'][
                ['hospitalization_id', 'recorded_dttm', 'vital_value']
            ].rename(columns={'vital_value': 'hr'})
            
            sbp = result[result['vital_category'] == 'Systolic blood pressure'][
                ['hospitalization_id', 'recorded_dttm', 'vital_value']
            ].rename(columns={'vital_value': 'sbp'})
            
            # Merge on hospitalization and approximate time
            shock_merged = pd.merge_asof(
                hr.sort_values('recorded_dttm'),
                sbp.sort_values('recorded_dttm'),
                on='recorded_dttm',
                by='hospitalization_id',
                tolerance=pd.Timedelta(minutes=5)
            )
            
            # Calculate shock index
            shock_merged['shock_index'] = shock_merged['hr'] / shock_merged['sbp']
            
            # Create result
            result = shock_merged[['hospitalization_id', 'recorded_dttm', 'shock_index']].dropna()
            result['vital_category'] = 'Shock index'
            result['vital_value'] = result['shock_index']
            result = result.drop('shock_index', axis=1)
            
        # Custom definition handling
        elif 'calculation' in definition:
            result = self._apply_timepoint_calculation(
                result, variable_name, definition
            )
            
        return result
    
    def _apply_custom_definition(self, 
                               base_df: pd.DataFrame,
                               variable_name: str,
                               variable_type: str,
                               definition: Dict[str, Any],
                               level: str) -> pd.DataFrame:
        """Apply custom variable definition"""
        
        source_table = definition['source_table']
        source_df = self.tables[source_table].copy()
        
        # Apply filters if specified
        if 'filters' in definition:
            for field, value in definition['filters'].items():
                if isinstance(value, list):
                    source_df = source_df[source_df[field].isin(value)]
                else:
                    source_df = source_df[source_df[field] == value]
        
        # Apply aggregation
        if 'aggregation' in definition:
            agg_method = definition['aggregation']['method']
            agg_field = definition['aggregation']['field']
            group_by = definition['aggregation'].get('group_by', 'hospitalization_id')
            
            if level == 'patient':
                # Need to join with hospitalization to get patient_id
                if 'patient_id' not in source_df.columns:
                    source_df = source_df.merge(
                        self.tables['hospitalization'][['hospitalization_id', 'patient_id']],
                        on='hospitalization_id'
                    )
                group_by = 'patient_id'
            
            # Perform aggregation
            if agg_method == 'count':
                agg_result = source_df.groupby(group_by).size()
            elif agg_method == 'sum':
                agg_result = source_df.groupby(group_by)[agg_field].sum()
            elif agg_method == 'mean':
                agg_result = source_df.groupby(group_by)[agg_field].mean()
            elif agg_method == 'max':
                agg_result = source_df.groupby(group_by)[agg_field].max()
            elif agg_method == 'min':
                agg_result = source_df.groupby(group_by)[agg_field].min()
            elif agg_method == 'first':
                agg_result = source_df.groupby(group_by)[agg_field].first()
            elif agg_method == 'last':
                agg_result = source_df.groupby(group_by)[agg_field].last()
            
            agg_result = agg_result.reset_index(name=variable_name)
            
            # Merge back to base dataframe
            join_key = 'patient_id' if level == 'patient' else 'hospitalization_id'
            base_df = base_df.merge(agg_result, on=join_key, how='left')
        
        # Apply transformation if specified
        if 'transformation' in definition:
            transform = definition['transformation']
            if transform['type'] == 'binary':
                threshold = transform.get('threshold', 0)
                operator = transform.get('operator', '>')
                
                if operator == '>':
                    base_df[variable_name] = (base_df[variable_name] > threshold).astype(int)
                elif operator == '>=':
                    base_df[variable_name] = (base_df[variable_name] >= threshold).astype(int)
                elif operator == '<':
                    base_df[variable_name] = (base_df[variable_name] < threshold).astype(int)
                elif operator == '<=':
                    base_df[variable_name] = (base_df[variable_name] <= threshold).astype(int)
                elif operator == '==':
                    base_df[variable_name] = (base_df[variable_name] == threshold).astype(int)
                elif operator == 'in':
                    base_df[variable_name] = base_df[variable_name].isin(transform['values']).astype(int)
        
        return base_df
    
    def _apply_timepoint_calculation(self,
                                   df: pd.DataFrame,
                                   variable_name: str,
                                   definition: Dict[str, Any]) -> pd.DataFrame:
        """Apply calculation for timepoint variables"""
        
        calc = definition['calculation']
        
        if calc['type'] == 'ratio':
            # Calculate ratio of two variables
            numerator = df[df[definition['category_field']] == calc['numerator']]
            denominator = df[df[definition['category_field']] == calc['denominator']]
            
            # Merge on time and hospitalization
            merged = pd.merge_asof(
                numerator.sort_values(definition['time_field']),
                denominator.sort_values(definition['time_field']),
                on=definition['time_field'],
                by='hospitalization_id',
                tolerance=pd.Timedelta(minutes=calc.get('tolerance_minutes', 5)),
                suffixes=('_num', '_den')
            )
            
            # Calculate ratio
            value_field = definition.get('value_field', 'value')
            merged[variable_name] = merged[f'{value_field}_num'] / merged[f'{value_field}_den']
            
            # Format result
            result = merged[['hospitalization_id', definition['time_field'], variable_name]].dropna()
            result[definition['category_field']] = variable_name
            result = result.rename(columns={variable_name: value_field})
            
        elif calc['type'] == 'difference':
            # Calculate difference between two variables
            var1 = df[df[definition['category_field']] == calc['var1']]
            var2 = df[df[definition['category_field']] == calc['var2']]
            
            # Merge on time and hospitalization
            merged = pd.merge_asof(
                var1.sort_values(definition['time_field']),
                var2.sort_values(definition['time_field']),
                on=definition['time_field'],
                by='hospitalization_id',
                tolerance=pd.Timedelta(minutes=calc.get('tolerance_minutes', 5)),
                suffixes=('_1', '_2')
            )
            
            # Calculate difference
            value_field = definition.get('value_field', 'value')
            merged[variable_name] = merged[f'{value_field}_1'] - merged[f'{value_field}_2']
            
            # Format result
            result = merged[['hospitalization_id', definition['time_field'], variable_name]].dropna()
            result[definition['category_field']] = variable_name
            result = result.rename(columns={variable_name: value_field})
            
        return result
    
    def get_common_derived_variables(self) -> Dict[str, Dict[str, Any]]:
        """Return dictionary of common derived variables and their definitions"""
        
        return {
            "hospitalization_level": {
                "mortality": {
                    "description": "In-hospital mortality (binary: 0=survived, 1=died)",
                    "type": "binary",
                    "level": "hospitalization"
                },
                "los_days": {
                    "description": "Hospital length of stay in days",
                    "type": "numeric",
                    "level": "hospitalization"
                },
                "icu_los_days": {
                    "description": "ICU length of stay in days",
                    "type": "numeric",
                    "level": "hospitalization"
                },
                "mechanical_ventilation": {
                    "description": "Received mechanical ventilation (binary)",
                    "type": "binary",
                    "level": "hospitalization"
                },
                "ventilator_days": {
                    "description": "Total days on mechanical ventilation",
                    "type": "numeric",
                    "level": "hospitalization"
                },
                "aki_stage": {
                    "description": "Acute kidney injury stage (0=none, 1-3=stage)",
                    "type": "categorical",
                    "level": "hospitalization"
                },
                "max_sofa_score": {
                    "description": "Maximum SOFA score during hospitalization",
                    "type": "numeric",
                    "level": "hospitalization"
                }
            },
            "patient_level": {
                "total_hospitalizations": {
                    "description": "Total number of hospitalizations",
                    "type": "numeric",
                    "level": "patient"
                },
                "total_icu_days": {
                    "description": "Total ICU days across all hospitalizations",
                    "type": "numeric",
                    "level": "patient"
                },
                "ever_ventilated": {
                    "description": "Ever received mechanical ventilation (binary)",
                    "type": "binary",
                    "level": "patient"
                },
                "mortality_any_admission": {
                    "description": "Died in any admission (binary)",
                    "type": "binary",
                    "level": "patient"
                }
            },
            "timepoint_level": {
                "map": {
                    "description": "Mean arterial pressure calculated from SBP and DBP",
                    "type": "numeric",
                    "level": "timepoint",
                    "source_table": "vitals"
                },
                "shock_index": {
                    "description": "Shock index (HR/SBP)",
                    "type": "numeric",
                    "level": "timepoint",
                    "source_table": "vitals"
                }
            }
        }