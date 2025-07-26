"""
Dynamic Cohort Builder Tool for CLIF MCP Server
Creates patient cohorts based on clinical criteria using dynamic schema
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import hashlib
import logging

from .schema_loader import SchemaLoader

logger = logging.getLogger(__name__)


class CohortBuilder:
    def __init__(self, schema_path: str, data_path: str):
        """
        Initialize cohort builder with dynamic schema
        
        Args:
            schema_path: Path to schema configuration JSON
            data_path: Base directory containing data files
        """
        self.data_path = Path(data_path)
        self.schema_loader = SchemaLoader(schema_path, data_path)
        self.cohorts_dir = self.data_path.parent / "cohorts"
        self.cohorts_dir.mkdir(exist_ok=True)
        
        # Discover available files
        self.available_files = self.schema_loader.discover_files()
        
    def build_cohort(self, criteria: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
        """Build cohort based on specified criteria"""
        
        # Start with hospitalizations if available, otherwise use patient table
        base_table = 'hospitalization' if 'hospitalization' in self.available_files else 'patient'
        cohort = self.schema_loader.load_table(base_table)
        
        if cohort is None:
            raise ValueError(f"Could not load base table '{base_table}' for cohort building")
            
        cohort_filters = []
        
        # Parse datetime columns dynamically based on schema
        table_info = self.schema_loader.get_table_info(base_table)
        if table_info:
            for col_name, col_info in table_info.get('columns', {}).items():
                if col_info.get('type') == 'datetime' and col_name in cohort.columns:
                    cohort[col_name] = pd.to_datetime(cohort[col_name], errors='coerce')
        
        # Age filter (flexible column names)
        if 'age_range' in criteria:
            age_col = self._find_age_column(cohort)
            if age_col:
                age_min = criteria['age_range'].get('min', 0)
                age_max = criteria['age_range'].get('max', 120)
                mask = (cohort[age_col] >= age_min) & (cohort[age_col] <= age_max)
                cohort = cohort[mask]
                cohort_filters.append(f"age_{age_min}_{age_max}")
                logger.info(f"Applied age filter using column '{age_col}': {age_min}-{age_max}")
        
        # ICU LOS filter
        if 'icu_los_range' in criteria and 'adt' in self.available_files:
            icu_los = self._calculate_icu_los(cohort)
            if icu_los is not None:
                los_min = criteria['icu_los_range'].get('min', 0)
                los_max = criteria['icu_los_range'].get('max', float('inf'))
                
                valid_ids = icu_los[
                    (icu_los['icu_days'] >= los_min) & 
                    (icu_los['icu_days'] <= los_max)
                ]['hospitalization_id']
                
                cohort = cohort[cohort['hospitalization_id'].isin(valid_ids)]
                cohort_filters.append(f"icu_los_{los_min}_{los_max}")
        
        # Mechanical ventilation filter
        if criteria.get('require_mechanical_ventilation', False) and 'respiratory_support' in self.available_files:
            vent_ids = self._get_ventilated_patients()
            if vent_ids is not None:
                id_col = 'hospitalization_id' if 'hospitalization_id' in cohort.columns else 'patient_id'
                cohort = cohort[cohort[id_col].isin(vent_ids)]
                cohort_filters.append("ventilated")
        
        # Medication filters
        if 'medications' in criteria and criteria['medications'] and 'medication_administration' in self.available_files:
            med_ids = self._filter_by_medications(criteria['medications'])
            if med_ids is not None:
                id_col = 'hospitalization_id' if 'hospitalization_id' in cohort.columns else 'patient_id'
                cohort = cohort[cohort[id_col].isin(med_ids)]
                cohort_filters.append(f"meds_{'_'.join(criteria['medications'][:3])}")
        
        # Lab-based criteria
        if 'lab_criteria' in criteria and criteria['lab_criteria'] and 'labs' in self.available_files:
            lab_ids = self._filter_by_lab_criteria(criteria['lab_criteria'])
            if lab_ids is not None:
                id_col = 'hospitalization_id' if 'hospitalization_id' in cohort.columns else 'patient_id'
                cohort = cohort[cohort[id_col].isin(lab_ids)]
                cohort_filters.append("lab_criteria")
        
        # Custom table filters
        if 'custom_filters' in criteria:
            cohort = self._apply_custom_filters(cohort, criteria['custom_filters'])
            cohort_filters.append("custom")
        
        # Generate cohort ID
        cohort_id = self._generate_cohort_id(criteria, len(cohort))
        
        logger.info(f"Built cohort '{cohort_id}' with {len(cohort)} records using filters: {cohort_filters}")
        
        return cohort, cohort_id
    
    def _find_age_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find age-related column in dataframe"""
        age_patterns = ['age', 'anchor_age', 'admission_age', 'age_at_admission']
        for pattern in age_patterns:
            for col in df.columns:
                if pattern in col.lower():
                    return col
        return None
    
    def _calculate_icu_los(self, cohort: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate ICU length of stay from ADT events"""
        adt = self.schema_loader.load_table('adt')
        if adt is None:
            return None
            
        # Parse datetime columns
        time_cols = ['in_time', 'out_time', 'in_dttm', 'out_dttm']
        for col in time_cols:
            if col in adt.columns:
                adt[col] = pd.to_datetime(adt[col], errors='coerce')
        
        # Filter for ICU locations
        icu_patterns = ['ICU', 'MICU', 'SICU', 'CCU', 'CVICU', 'NICU', 'PICU']
        location_col = 'location_category' if 'location_category' in adt.columns else 'location'
        icu_mask = adt[location_col].str.upper().str.contains('|'.join(icu_patterns), na=False)
        icu_adt = adt[icu_mask].copy()
        
        # Calculate LOS using available time columns
        out_col = 'out_dttm' if 'out_dttm' in icu_adt.columns else 'out_time'
        in_col = 'in_dttm' if 'in_dttm' in icu_adt.columns else 'in_time'
        
        if out_col in icu_adt.columns and in_col in icu_adt.columns:
            icu_adt['los_hours'] = (icu_adt[out_col] - icu_adt[in_col]).dt.total_seconds() / 3600
        else:
            return None
        
        # Aggregate by hospitalization
        id_col = 'hospitalization_id' if 'hospitalization_id' in icu_adt.columns else 'patient_id'
        icu_los = icu_adt.groupby(id_col).agg({
            'los_hours': 'sum'
        }).reset_index()
        icu_los['icu_days'] = icu_los['los_hours'] / 24
        
        return icu_los
    
    def _get_ventilated_patients(self) -> Optional[set]:
        """Get patients who received mechanical ventilation"""
        resp_support = self.schema_loader.load_table('respiratory_support')
        if resp_support is None:
            return None
            
        # Filter for mechanical ventilation devices
        vent_patterns = ['Invasive', 'Mechanical', 'Ventilator', 'ETT', 'Trach']
        device_col = 'device' if 'device' in resp_support.columns else 'support_device'
        
        if device_col in resp_support.columns:
            vent_mask = resp_support[device_col].str.contains('|'.join(vent_patterns), case=False, na=False)
            vent_patients = resp_support[vent_mask]
            
            id_col = 'hospitalization_id' if 'hospitalization_id' in vent_patients.columns else 'patient_id'
            return set(vent_patients[id_col].unique())
        
        return None
    
    def _filter_by_medications(self, medications: List[str]) -> Optional[set]:
        """Filter patients who received specific medications"""
        med_admin = self.schema_loader.load_table('medication_administration')
        if med_admin is None:
            return None
            
        # Find medication column
        med_col = 'medication' if 'medication' in med_admin.columns else 'drug_name'
        if med_col not in med_admin.columns:
            return None
            
        # Filter for requested medications
        pattern = '|'.join(medications)
        mask = med_admin[med_col].str.contains(pattern, case=False, na=False)
        filtered_meds = med_admin[mask]
        
        id_col = 'hospitalization_id' if 'hospitalization_id' in filtered_meds.columns else 'patient_id'
        return set(filtered_meds[id_col].unique())
    
    def _filter_by_lab_criteria(self, lab_criteria: List[Dict[str, Any]]) -> Optional[set]:
        """Filter patients based on lab criteria"""
        labs = self.schema_loader.load_table('labs')
        if labs is None:
            return None
            
        matching_ids = set()
        
        for criterion in lab_criteria:
            test_name = criterion.get('test_name')
            min_value = criterion.get('min_value')
            max_value = criterion.get('max_value')
            
            if not test_name:
                continue
                
            # Find test name column
            name_col = 'test_name' if 'test_name' in labs.columns else 'lab_name'
            result_col = 'test_result' if 'test_result' in labs.columns else 'result_value'
            
            if name_col not in labs.columns or result_col not in labs.columns:
                continue
                
            # Filter for specific test
            test_mask = labs[name_col].str.contains(test_name, case=False, na=False)
            test_labs = labs[test_mask]
            
            # Apply value filters
            if min_value is not None:
                test_labs = test_labs[test_labs[result_col] >= min_value]
            if max_value is not None:
                test_labs = test_labs[test_labs[result_col] <= max_value]
                
            id_col = 'hospitalization_id' if 'hospitalization_id' in test_labs.columns else 'patient_id'
            matching_ids.update(test_labs[id_col].unique())
        
        return matching_ids if matching_ids else None
    
    def _apply_custom_filters(self, cohort: pd.DataFrame, custom_filters: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply custom filters based on any table and column"""
        for filter_config in custom_filters:
            table_name = filter_config.get('table')
            column = filter_config.get('column')
            operator = filter_config.get('operator', 'equals')
            value = filter_config.get('value')
            
            if not all([table_name, column, value is not None]):
                continue
                
            # Load the custom table
            custom_table = self.schema_loader.load_table(table_name)
            if custom_table is None or column not in custom_table.columns:
                continue
                
            # Apply the filter operation
            if operator == 'equals':
                mask = custom_table[column] == value
            elif operator == 'contains':
                mask = custom_table[column].str.contains(value, case=False, na=False)
            elif operator == 'greater_than':
                mask = custom_table[column] > value
            elif operator == 'less_than':
                mask = custom_table[column] < value
            elif operator == 'in':
                mask = custom_table[column].isin(value)
            else:
                continue
                
            # Get matching IDs
            filtered_table = custom_table[mask]
            
            # Find common ID column
            for id_col in ['hospitalization_id', 'patient_id']:
                if id_col in filtered_table.columns and id_col in cohort.columns:
                    matching_ids = set(filtered_table[id_col].unique())
                    cohort = cohort[cohort[id_col].isin(matching_ids)]
                    break
        
        return cohort
    
    def _generate_cohort_id(self, criteria: Dict[str, Any], size: int) -> str:
        """Generate unique cohort ID"""
        criteria_str = json.dumps(criteria, sort_keys=True)
        hash_val = hashlib.md5(criteria_str.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"cohort_{timestamp}_{hash_val}_n{size}"
    
    def save_cohort(self, cohort: pd.DataFrame, cohort_id: str, metadata: Optional[Dict] = None) -> str:
        """Save cohort to file with metadata"""
        cohort_file = self.cohorts_dir / f"{cohort_id}.csv"
        cohort.to_csv(cohort_file, index=False)
        
        # Save metadata
        if metadata:
            metadata_file = self.cohorts_dir / f"{cohort_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        return str(cohort_file)
    
    def get_schema_summary(self) -> str:
        """Get summary of available data based on schema"""
        return self.schema_loader.get_schema_summary()