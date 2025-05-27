"""
Flexible Cohort Builder for CLIF MCP Server
Allows filtering on any variable in any table with complex criteria
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import hashlib
import operator


class FlexibleCohortBuilder:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.tables = self._load_tables()
        self.cohorts_dir = self.data_path.parent / "cohorts"
        self.cohorts_dir.mkdir(exist_ok=True)
        
        # Define operators for filtering
        self.operators = {
            'eq': operator.eq,
            'ne': operator.ne,
            'gt': operator.gt,
            'gte': operator.ge,
            'lt': operator.lt,
            'lte': operator.le,
            'in': lambda x, y: x in y,
            'not_in': lambda x, y: x not in y,
            'contains': lambda x, y: y in str(x),
            'not_contains': lambda x, y: y not in str(x),
            'between': lambda x, y: y[0] <= x <= y[1],
            'is_null': lambda x, y: pd.isna(x),
            'not_null': lambda x, y: pd.notna(x)
        }
        
    def _load_tables(self) -> Dict[str, pd.DataFrame]:
        """Load all CLIF tables"""
        tables = {}
        
        for csv_file in self.data_path.glob("*.csv"):
            table_name = csv_file.stem
            df = pd.read_csv(csv_file)
            
            # Parse datetime columns
            for col in df.columns:
                if 'dttm' in col or 'date' in col:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            tables[table_name] = df
            
        return tables
    
    def build_cohort_flexible(self, filters: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, str]:
        """
        Build cohort with flexible filtering including time windows
        
        filters: List of filter dictionaries, each containing:
            - table: Table name (e.g., 'hospitalization', 'labs', 'vitals')
            - variable: Variable name 
            - operator: Comparison operator ('eq', 'gt', 'lt', 'between', 'in', etc.)
            - value: Value(s) to compare against
            - aggregation: (optional) For time-series data: 'max', 'min', 'mean', 'any', 'all'
            - time_window: (optional) Time window specification:
                - 'first_24h', 'first_48h', 'first_72h': From admission
                - 'last_24h', 'last_48h': Before discharge
                - {'start': 0, 'end': 24}: Custom hours from admission
                - {'start': -48, 'end': 0}: Custom hours before discharge (negative = from end)
                - 'icu': During ICU stay only
                - 'pre_icu': Before first ICU admission
                - 'post_icu': After last ICU discharge
            
        Examples:
            {'table': 'labs', 'variable': 'lab_value', 'operator': 'gt', 'value': 1.5, 
             'lab_category': 'Creatinine', 'time_window': 'first_24h', 'aggregation': 'any'}
            {'table': 'vitals', 'variable': 'vital_value', 'operator': 'lt', 'value': 90,
             'vital_category': 'Systolic Blood Pressure', 'time_window': {'start': 0, 'end': 6}}
        """
        
        # Start with all hospitalizations
        cohort = self.tables['hospitalization'].copy()
        filter_descriptions = []
        
        # Process each filter
        for filter_spec in filters:
            table_name = filter_spec.get('table', 'hospitalization')
            variable = filter_spec['variable']
            op = filter_spec.get('operator', 'eq')
            value = filter_spec['value']
            aggregation = filter_spec.get('aggregation')
            
            # Handle filters on hospitalization table directly
            if table_name == 'hospitalization':
                cohort = self._apply_direct_filter(cohort, variable, op, value)
                filter_descriptions.append(f"{variable} {op} {value}")
                
            # Handle filters on patient table (merge required)
            elif table_name == 'patient':
                patient_df = self.tables['patient']
                # Apply filter to patient table
                filtered_patients = self._apply_direct_filter(patient_df, variable, op, value)
                # Keep only hospitalizations for filtered patients
                cohort = cohort[cohort['patient_id'].isin(filtered_patients['patient_id'])]
                filter_descriptions.append(f"patient.{variable} {op} {value}")
                
            # Handle time-series tables (vitals, labs, etc.)
            elif table_name in ['vitals', 'labs', 'respiratory_support', 'medication_administration', 
                               'patient_assessments', 'adt']:
                hosp_ids = self._filter_by_timeseries(table_name, filter_spec)
                cohort = cohort[cohort['hospitalization_id'].isin(hosp_ids)]
                
                agg_str = f" ({aggregation})" if aggregation else ""
                filter_descriptions.append(f"{table_name}.{variable}{agg_str} {op} {value}")
                
            # Handle derived variables
            elif table_name == 'derived':
                cohort = self._apply_derived_filter(cohort, variable, op, value)
                filter_descriptions.append(f"derived.{variable} {op} {value}")
        
        # Generate cohort ID
        cohort_id = self._generate_cohort_id(filters, len(cohort))
        
        # Store filter criteria with cohort
        cohort.attrs = {'filters': filters, 'description': ' AND '.join(filter_descriptions)}
        
        return cohort, cohort_id
    
    def _apply_direct_filter(self, df: pd.DataFrame, variable: str, op: str, value: Any) -> pd.DataFrame:
        """Apply filter directly to a dataframe"""
        if variable not in df.columns:
            raise ValueError(f"Variable '{variable}' not found in table. Available: {list(df.columns)}")
        
        if op in self.operators:
            if op in ['is_null', 'not_null']:
                mask = df[variable].apply(lambda x: self.operators[op](x, None))
            else:
                mask = df[variable].apply(lambda x: self.operators[op](x, value))
            return df[mask]
        else:
            raise ValueError(f"Unknown operator: {op}")
    
    def _filter_by_timeseries(self, table_name: str, filter_spec: Dict[str, Any]) -> List[str]:
        """Filter hospitalizations based on time-series data with time window support"""
        table_df = self.tables[table_name]
        variable = filter_spec['variable']
        op = filter_spec.get('operator', 'eq')
        value = filter_spec['value']
        aggregation = filter_spec.get('aggregation', 'any')
        time_window = filter_spec.get('time_window')
        
        # Apply additional filters first (e.g., lab_category, vital_category)
        for key, val in filter_spec.items():
            if key not in ['table', 'variable', 'operator', 'value', 'aggregation', 'time_window'] and key in table_df.columns:
                table_df = table_df[table_df[key] == val]
        
        # If time window specified, apply it
        if time_window:
            table_df = self._apply_time_window(table_df, table_name, time_window)
        
        # Group by hospitalization and apply aggregation
        if aggregation == 'max':
            grouped = table_df.groupby('hospitalization_id')[variable].max()
        elif aggregation == 'min':
            grouped = table_df.groupby('hospitalization_id')[variable].min()
        elif aggregation == 'mean':
            grouped = table_df.groupby('hospitalization_id')[variable].mean()
        elif aggregation == 'count':
            grouped = table_df.groupby('hospitalization_id')[variable].count()
        elif aggregation == 'any':
            # For categorical variables - check if any row matches
            if op == 'eq':
                grouped = table_df[table_df[variable] == value].groupby('hospitalization_id').size() > 0
            else:
                mask = table_df[variable].apply(lambda x: self.operators[op](x, value))
                grouped = table_df[mask].groupby('hospitalization_id').size() > 0
        elif aggregation == 'all':
            # All values must match
            if op == 'eq':
                grouped = table_df.groupby('hospitalization_id')[variable].apply(lambda x: (x == value).all())
            else:
                grouped = table_df.groupby('hospitalization_id')[variable].apply(
                    lambda x: x.apply(lambda v: self.operators[op](v, value)).all()
                )
        else:
            # First/last value
            if aggregation == 'first':
                grouped = table_df.sort_values('recorded_dttm').groupby('hospitalization_id')[variable].first()
            else:
                grouped = table_df.sort_values('recorded_dttm').groupby('hospitalization_id')[variable].last()
        
        # Apply operator to aggregated values
        if aggregation not in ['any', 'all']:
            mask = grouped.apply(lambda x: self.operators[op](x, value))
            return list(grouped[mask].index)
        else:
            return list(grouped[grouped].index)
    
    def _apply_derived_filter(self, cohort: pd.DataFrame, variable: str, op: str, value: Any) -> pd.DataFrame:
        """Apply filters for derived variables"""
        
        # Calculate derived variables on demand
        if variable == 'mortality':
            cohort['mortality'] = (cohort['discharge_disposition'] == 'Expired').astype(int)
        elif variable == 'los_days':
            cohort['los_days'] = (cohort['discharge_dttm'] - cohort['admission_dttm']).dt.total_seconds() / 86400
        elif variable == 'ventilation':
            resp_df = self.tables.get('respiratory_support')
            if resp_df is not None:
                vent_ids = resp_df[resp_df['device_category'] == 'Invasive Mechanical Ventilation']['hospitalization_id'].unique()
                cohort['ventilation'] = cohort['hospitalization_id'].isin(vent_ids).astype(int)
        elif variable == 'vasopressor_use':
            med_df = self.tables.get('medication_administration')
            if med_df is not None:
                vaso_ids = med_df[med_df['medication_category'] == 'Vasopressor']['hospitalization_id'].unique()
                cohort['vasopressor_use'] = cohort['hospitalization_id'].isin(vaso_ids).astype(int)
        elif variable == 'icu_admission':
            adt_df = self.tables.get('adt')
            if adt_df is not None:
                icu_ids = adt_df[adt_df['location_category'] == 'ICU']['hospitalization_id'].unique()
                cohort['icu_admission'] = cohort['hospitalization_id'].isin(icu_ids).astype(int)
        else:
            raise ValueError(f"Unknown derived variable: {variable}")
        
        # Apply filter
        return self._apply_direct_filter(cohort, variable, op, value)
    
    def _apply_time_window(self, df: pd.DataFrame, table_name: str, time_window: Union[str, Dict]) -> pd.DataFrame:
        """Apply time window filtering to time-series data"""
        
        # Get hospitalization times
        hosp_df = self.tables['hospitalization'][['hospitalization_id', 'admission_dttm', 'discharge_dttm']]
        
        # Determine time column based on table
        time_col_map = {
            'vitals': 'recorded_dttm',
            'labs': 'lab_result_dttm',
            'medication_administration': 'admin_dttm',
            'patient_assessments': 'assessment_dttm',
            'respiratory_support': 'start_dttm',
            'adt': 'in_dttm',
            'position': 'start_dttm'
        }
        
        time_col = time_col_map.get(table_name, 'recorded_dttm')
        
        # Merge with hospitalization times
        df = df.merge(hosp_df, on='hospitalization_id', how='left')
        
        # Calculate hours from admission and before discharge
        df['hours_from_admission'] = (df[time_col] - df['admission_dttm']).dt.total_seconds() / 3600
        df['hours_before_discharge'] = (df['discharge_dttm'] - df[time_col]).dt.total_seconds() / 3600
        
        # Apply time window
        if isinstance(time_window, str):
            if time_window == 'first_24h':
                mask = (df['hours_from_admission'] >= 0) & (df['hours_from_admission'] <= 24)
            elif time_window == 'first_48h':
                mask = (df['hours_from_admission'] >= 0) & (df['hours_from_admission'] <= 48)
            elif time_window == 'first_72h':
                mask = (df['hours_from_admission'] >= 0) & (df['hours_from_admission'] <= 72)
            elif time_window == 'last_24h':
                mask = (df['hours_before_discharge'] >= 0) & (df['hours_before_discharge'] <= 24)
            elif time_window == 'last_48h':
                mask = (df['hours_before_discharge'] >= 0) & (df['hours_before_discharge'] <= 48)
            elif time_window == 'icu':
                # During ICU stay
                mask = self._filter_during_icu(df, time_col)
            elif time_window == 'pre_icu':
                # Before first ICU admission
                mask = self._filter_pre_icu(df, time_col)
            elif time_window == 'post_icu':
                # After last ICU discharge
                mask = self._filter_post_icu(df, time_col)
            else:
                raise ValueError(f"Unknown time window: {time_window}")
                
        elif isinstance(time_window, dict):
            # Custom time window
            start_hours = time_window.get('start', 0)
            end_hours = time_window.get('end', float('inf'))
            
            if start_hours >= 0 and end_hours >= 0:
                # From admission
                mask = (df['hours_from_admission'] >= start_hours) & (df['hours_from_admission'] <= end_hours)
            elif start_hours < 0 and end_hours <= 0:
                # Before discharge (negative hours)
                mask = (df['hours_before_discharge'] >= -end_hours) & (df['hours_before_discharge'] <= -start_hours)
            else:
                raise ValueError("Time window cannot span both admission and discharge")
        else:
            raise ValueError(f"Invalid time window type: {type(time_window)}")
        
        # Apply mask and drop helper columns
        filtered_df = df[mask].drop(columns=['admission_dttm', 'discharge_dttm', 
                                            'hours_from_admission', 'hours_before_discharge'])
        
        return filtered_df
    
    def _filter_during_icu(self, df: pd.DataFrame, time_col: str) -> pd.Series:
        """Filter for events during ICU stay"""
        adt_df = self.tables.get('adt')
        if adt_df is None:
            return pd.Series([False] * len(df))
        
        # Get ICU periods
        icu_periods = adt_df[adt_df['location_category'] == 'ICU'][
            ['hospitalization_id', 'in_dttm', 'out_dttm']
        ]
        
        # Check if each event falls within any ICU period
        mask = pd.Series([False] * len(df))
        
        for idx, row in df.iterrows():
            hosp_id = row['hospitalization_id']
            event_time = row[time_col]
            
            # Get ICU periods for this hospitalization
            hosp_icu = icu_periods[icu_periods['hospitalization_id'] == hosp_id]
            
            # Check if event time falls within any ICU period
            for _, icu_period in hosp_icu.iterrows():
                if icu_period['in_dttm'] <= event_time <= icu_period['out_dttm']:
                    mask.iloc[idx] = True
                    break
        
        return mask
    
    def _filter_pre_icu(self, df: pd.DataFrame, time_col: str) -> pd.Series:
        """Filter for events before first ICU admission"""
        adt_df = self.tables.get('adt')
        if adt_df is None:
            return pd.Series([True] * len(df))
        
        # Get first ICU admission for each hospitalization
        first_icu = adt_df[adt_df['location_category'] == 'ICU'].groupby(
            'hospitalization_id')['in_dttm'].min().reset_index()
        first_icu.columns = ['hospitalization_id', 'first_icu_time']
        
        # Merge with events
        df_with_icu = df.merge(first_icu, on='hospitalization_id', how='left')
        
        # Events before first ICU (or no ICU)
        mask = df_with_icu['first_icu_time'].isna() | (df[time_col] < df_with_icu['first_icu_time'])
        
        return mask
    
    def _filter_post_icu(self, df: pd.DataFrame, time_col: str) -> pd.Series:
        """Filter for events after last ICU discharge"""
        adt_df = self.tables.get('adt')
        if adt_df is None:
            return pd.Series([False] * len(df))
        
        # Get last ICU discharge for each hospitalization
        last_icu = adt_df[adt_df['location_category'] == 'ICU'].groupby(
            'hospitalization_id')['out_dttm'].max().reset_index()
        last_icu.columns = ['hospitalization_id', 'last_icu_time']
        
        # Merge with events
        df_with_icu = df.merge(last_icu, on='hospitalization_id', how='left')
        
        # Events after last ICU
        mask = df_with_icu['last_icu_time'].notna() & (df[time_col] > df_with_icu['last_icu_time'])
        
        return mask
    
    def _generate_cohort_id(self, filters: List[Dict], size: int) -> str:
        """Generate unique cohort ID"""
        # Create a hash of the filter criteria
        filter_str = json.dumps(filters, sort_keys=True)
        filter_hash = hashlib.md5(filter_str.encode()).hexdigest()[:8]
        
        # Create readable ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        cohort_id = f"cohort_{timestamp}_{filter_hash}_n{size}"
        
        return cohort_id
    
    def get_filter_suggestions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Suggest common filter patterns including time windows"""
        suggestions = {
            "demographics": [
                {"table": "hospitalization", "variable": "age_at_admission", "operator": "gte", "value": 65},
                {"table": "patient", "variable": "sex", "operator": "eq", "value": "Female"},
                {"table": "patient", "variable": "race", "operator": "in", "value": ["White", "Black or African American"]}
            ],
            "clinical_status": [
                {"table": "derived", "variable": "ventilation", "operator": "eq", "value": 1},
                {"table": "derived", "variable": "mortality", "operator": "eq", "value": 0},
                {"table": "hospitalization", "variable": "admission_source", "operator": "eq", "value": "Emergency Department"}
            ],
            "lab_values": [
                {"table": "labs", "variable": "lab_value", "operator": "gt", "value": 2.0, 
                 "lab_category": "Creatinine", "aggregation": "max"},
                {"table": "labs", "variable": "lab_value", "operator": "gt", "value": 4.0,
                 "lab_category": "Lactate", "aggregation": "any"}
            ],
            "vital_signs": [
                {"table": "vitals", "variable": "vital_value", "operator": "lt", "value": 90,
                 "vital_category": "Systolic Blood Pressure", "aggregation": "min"},
                {"table": "vitals", "variable": "vital_value", "operator": "gt", "value": 38.5,
                 "vital_category": "Temperature", "aggregation": "max"}
            ],
            "medications": [
                {"table": "medication_administration", "variable": "medication_category", 
                 "operator": "eq", "value": "Vasopressor", "aggregation": "any"},
                {"table": "medication_administration", "variable": "medication_name",
                 "operator": "eq", "value": "Norepinephrine", "aggregation": "any"}
            ],
            "time_windowed": [
                # AKI in first 24 hours
                {"table": "labs", "variable": "lab_value", "operator": "gt", "value": 1.5,
                 "lab_category": "Creatinine", "time_window": "first_24h", "aggregation": "any"},
                # Hypotension in first 6 hours
                {"table": "vitals", "variable": "vital_value", "operator": "lt", "value": 90,
                 "vital_category": "Systolic Blood Pressure", "time_window": {"start": 0, "end": 6}, "aggregation": "any"},
                # Fever during ICU stay
                {"table": "vitals", "variable": "vital_value", "operator": "gt", "value": 38.5,
                 "vital_category": "Temperature", "time_window": "icu", "aggregation": "any"},
                # Delirium after ICU discharge
                {"table": "patient_assessments", "variable": "assessment_value", "operator": "eq", "value": 1,
                 "assessment_category": "CAM-ICU", "time_window": "post_icu", "aggregation": "any"},
                # Rising lactate in last 24h before death/discharge
                {"table": "labs", "variable": "lab_value", "operator": "gt", "value": 4.0,
                 "lab_category": "Lactate", "time_window": "last_24h", "aggregation": "max"}
            ],
            "complex": [
                # Early AKI (creatinine > 1.5 in first 24h)
                [
                    {"table": "labs", "variable": "lab_value", "operator": "gt", "value": 1.5,
                     "lab_category": "Creatinine", "time_window": "first_24h", "aggregation": "any"}
                ],
                # Early shock (hypotension + vasopressors in first 6h)
                [
                    {"table": "vitals", "variable": "vital_value", "operator": "lt", "value": 90,
                     "vital_category": "Systolic Blood Pressure", "time_window": {"start": 0, "end": 6}, "aggregation": "any"},
                    {"table": "medication_administration", "variable": "medication_category",
                     "operator": "eq", "value": "Vasopressor", "time_window": {"start": 0, "end": 6}, "aggregation": "any"}
                ],
                # Delayed intubation (ventilation after 24h but not in first 24h)
                [
                    {"table": "respiratory_support", "variable": "device_category",
                     "operator": "ne", "value": "Invasive Mechanical Ventilation", "time_window": "first_24h", "aggregation": "all"},
                    {"table": "respiratory_support", "variable": "device_category",
                     "operator": "eq", "value": "Invasive Mechanical Ventilation", "time_window": {"start": 24, "end": 168}, "aggregation": "any"}
                ]
            ]
        }
        return suggestions