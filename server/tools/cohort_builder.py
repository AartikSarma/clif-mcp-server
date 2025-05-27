"""
Cohort Builder Tool for CLIF MCP Server
Creates patient cohorts based on clinical criteria
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import hashlib


class CohortBuilder:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.tables = self._load_tables()
        self.cohorts_dir = self.data_path.parent / "cohorts"
        self.cohorts_dir.mkdir(exist_ok=True)
        
    def _load_tables(self) -> Dict[str, pd.DataFrame]:
        """Load required CLIF tables"""
        tables = {}
        required = ['patient', 'hospitalization', 'adt', 'respiratory_support', 
                   'medication_administration', 'labs', 'patient_assessments']
        
        for table in required:
            filepath = self.data_path / f"{table}.csv"
            if filepath.exists():
                df = pd.read_csv(filepath)
                # Parse datetime columns
                for col in df.columns:
                    if 'dttm' in col or 'date' in col:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                tables[table] = df
        
        return tables
    
    def build_cohort(self, criteria: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
        """Build cohort based on specified criteria"""
        
        # Start with all hospitalizations
        cohort = self.tables['hospitalization'].copy()
        cohort_filters = []
        
        # Age filter
        if 'age_range' in criteria:
            age_min = criteria['age_range'].get('min', 0)
            age_max = criteria['age_range'].get('max', 120)
            mask = (cohort['age_at_admission'] >= age_min) & (cohort['age_at_admission'] <= age_max)
            cohort = cohort[mask]
            cohort_filters.append(f"age_{age_min}_{age_max}")
        
        # ICU LOS filter
        if 'icu_los_range' in criteria:
            icu_los = self._calculate_icu_los(cohort)
            los_min = criteria['icu_los_range'].get('min', 0)
            los_max = criteria['icu_los_range'].get('max', float('inf'))
            
            valid_ids = icu_los[
                (icu_los['icu_days'] >= los_min) & 
                (icu_los['icu_days'] <= los_max)
            ]['hospitalization_id']
            
            cohort = cohort[cohort['hospitalization_id'].isin(valid_ids)]
            cohort_filters.append(f"icu_los_{los_min}_{los_max}")
        
        # Mechanical ventilation filter
        if criteria.get('require_mechanical_ventilation', False):
            vent_ids = self._get_ventilated_patients()
            cohort = cohort[cohort['hospitalization_id'].isin(vent_ids)]
            cohort_filters.append("ventilated")
        
        # Medication filters
        if 'medications' in criteria and criteria['medications']:
            med_ids = self._filter_by_medications(criteria['medications'])
            cohort = cohort[cohort['hospitalization_id'].isin(med_ids)]
            cohort_filters.append(f"meds_{'_'.join(criteria['medications'][:3])}")
        
        # Exclude readmissions
        if criteria.get('exclude_readmissions', False):
            cohort = self._exclude_readmissions(cohort)
            cohort_filters.append("no_readmit")
        
        # Additional filters can be added here:
        # - Specific diagnoses (when diagnosis table available)
        # - Procedures
        # - Lab value ranges
        # - Severity scores
        
        # Generate cohort ID
        cohort_id = self._generate_cohort_id(criteria, len(cohort))
        
        return cohort, cohort_id
    
    def _calculate_icu_los(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Calculate ICU length of stay for each hospitalization"""
        adt = self.tables['adt']
        
        icu_los = []
        for hosp_id in cohort['hospitalization_id']:
            icu_stays = adt[
                (adt['hospitalization_id'] == hosp_id) &
                (adt['location_category'] == 'ICU')
            ]
            
            if not icu_stays.empty:
                total_hours = 0
                for _, stay in icu_stays.iterrows():
                    duration = stay['out_dttm'] - stay['in_dttm']
                    total_hours += duration.total_seconds() / 3600
                
                icu_los.append({
                    'hospitalization_id': hosp_id,
                    'icu_days': total_hours / 24
                })
            else:
                icu_los.append({
                    'hospitalization_id': hosp_id,
                    'icu_days': 0
                })
        
        return pd.DataFrame(icu_los)
    
    def _get_ventilated_patients(self) -> List[str]:
        """Get hospitalization IDs for mechanically ventilated patients"""
        resp_support = self.tables['respiratory_support']
        
        vent_patients = resp_support[
            resp_support['device_category'] == 'Invasive Mechanical Ventilation'
        ]['hospitalization_id'].unique()
        
        return list(vent_patients)
    
    def _filter_by_medications(self, medications: List[str]) -> List[str]:
        """Filter hospitalizations by medication exposure"""
        meds = self.tables['medication_administration']
        
        # Convert medication names to lowercase for matching
        medications_lower = [m.lower() for m in medications]
        
        # Find hospitalizations with any of the specified medications
        mask = meds['medication_name'].str.lower().str.contains(
            '|'.join(medications_lower), na=False
        )
        
        return list(meds[mask]['hospitalization_id'].unique())
    
    def _exclude_readmissions(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Exclude readmissions, keeping only index admissions"""
        # Sort by patient and admission date
        cohort_sorted = cohort.sort_values(['patient_id', 'admission_dttm'])
        
        # Keep only first admission per patient
        index_admissions = cohort_sorted.groupby('patient_id').first().reset_index()
        
        return cohort[cohort['hospitalization_id'].isin(index_admissions['hospitalization_id'])]
    
    def _generate_cohort_id(self, criteria: Dict, size: int) -> str:
        """Generate unique cohort ID"""
        # Create a hash of the criteria
        criteria_str = json.dumps(criteria, sort_keys=True)
        criteria_hash = hashlib.md5(criteria_str.encode()).hexdigest()[:8]
        
        # Create readable ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        cohort_id = f"cohort_{timestamp}_{criteria_hash}_n{size}"
        
        return cohort_id
    
    def save_cohort(self, cohort_id: str, cohort: pd.DataFrame, criteria: Dict):
        """Save cohort definition and data"""
        # Save cohort data
        cohort_path = self.cohorts_dir / f"{cohort_id}.csv"
        cohort.to_csv(cohort_path, index=False)
        
        # Save cohort metadata
        metadata = {
            'cohort_id': cohort_id,
            'created': datetime.now().isoformat(),
            'size': len(cohort),
            'criteria': criteria,
            'unique_patients': cohort['patient_id'].nunique()
        }
        
        metadata_path = self.cohorts_dir / f"{cohort_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_cohort(self, cohort_id: str) -> pd.DataFrame:
        """Load previously saved cohort"""
        cohort_path = self.cohorts_dir / f"{cohort_id}.csv"
        if cohort_path.exists():
            return pd.read_csv(cohort_path)
        else:
            raise ValueError(f"Cohort {cohort_id} not found")
    
    def list_cohorts(self) -> List[Dict[str, Any]]:
        """List all saved cohorts with metadata"""
        cohorts = []
        
        for metadata_file in self.cohorts_dir.glob("*_metadata.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                cohorts.append(metadata)
        
        return sorted(cohorts, key=lambda x: x['created'], reverse=True)
    
    def get_cohort_characteristics(self, cohort: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary characteristics for a cohort"""
        
        # Merge with patient data
        patients = self.tables['patient']
        cohort_with_demo = cohort.merge(patients, on='patient_id', how='left')
        
        # Calculate ICU LOS
        icu_los = self._calculate_icu_los(cohort)
        
        # Get ventilation status
        vent_ids = self._get_ventilated_patients()
        n_ventilated = cohort['hospitalization_id'].isin(vent_ids).sum()
        
        characteristics = {
            'demographics': {
                'age_mean': float(cohort['age_at_admission'].mean()),
                'age_std': float(cohort['age_at_admission'].std()),
                'age_median': float(cohort['age_at_admission'].median()),
                'sex_distribution': cohort_with_demo['sex'].value_counts().to_dict(),
                'race_distribution': cohort_with_demo['race'].value_counts().to_dict()
            },
            'clinical': {
                'mortality_rate': float((cohort['discharge_disposition'] == 'Expired').mean()),
                'icu_los_mean': float(icu_los['icu_days'].mean()),
                'icu_los_median': float(icu_los['icu_days'].median()),
                'ventilation_rate': float(n_ventilated / len(cohort)),
                'admission_sources': cohort['admission_source'].value_counts().to_dict()
            },
            'cohort_info': {
                'total_hospitalizations': len(cohort),
                'unique_patients': cohort['patient_id'].nunique(),
                'date_range': {
                    'earliest': cohort['admission_dttm'].min().isoformat(),
                    'latest': cohort['discharge_dttm'].max().isoformat()
                }
            }
        }
        
        return characteristics