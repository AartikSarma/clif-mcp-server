"""
Data Explorer Tool for CLIF MCP Server
Provides schema exploration and data quality metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


class DataExplorer:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.tables = self._discover_tables()
        
    def _discover_tables(self) -> List[str]:
        """Discover available CLIF tables"""
        csv_files = list(self.data_path.glob("*.csv"))
        return [f.stem for f in csv_files]
    
    def explore_schema(self, table_name: Optional[str] = None) -> str:
        """Explore dataset schema and quality metrics"""
        
        if table_name:
            return self._explore_single_table(table_name)
        else:
            return self._explore_all_tables()
    
    def _explore_all_tables(self) -> str:
        """Provide overview of all tables"""
        output = []
        output.append("CLIF Dataset Overview")
        output.append("=" * 50)
        output.append(f"Data directory: {self.data_path}")
        output.append(f"Available tables: {len(self.tables)}\n")
        
        for table in sorted(self.tables):
            try:
                df = pd.read_csv(self.data_path / f"{table}.csv", nrows=5)
                output.append(f"{table.upper()}:")
                output.append(f"  Columns: {len(df.columns)}")
                output.append(f"  Sample columns: {', '.join(df.columns[:5])}")
                
                # Get row count efficiently
                with open(self.data_path / f"{table}.csv", 'r') as f:
                    row_count = sum(1 for line in f) - 1
                output.append(f"  Rows: {row_count:,}")
                
            except Exception as e:
                output.append(f"  Error reading table: {str(e)}")
            
            output.append("")
        
        return "\n".join(output)
    
    def _explore_single_table(self, table_name: str) -> str:
        """Detailed exploration of a single table"""
        
        if table_name not in self.tables:
            return f"Table '{table_name}' not found. Available tables: {', '.join(self.tables)}"
        
        try:
            # Load table
            df = pd.read_csv(self.data_path / f"{table_name}.csv")
            
            # Parse datetime columns
            for col in df.columns:
                if 'dttm' in col or 'date' in col:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            output = []
            output.append(f"Table: {table_name.upper()}")
            output.append("=" * 50)
            output.append(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
            
            # Column information
            output.append("COLUMNS:")
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100
                unique_count = df[col].nunique()
                
                output.append(f"  {col}:")
                output.append(f"    Type: {dtype}")
                output.append(f"    Missing: {null_count:,} ({null_pct:.1f}%)")
                output.append(f"    Unique values: {unique_count:,}")
                
                # Show sample values for categorical columns
                if dtype == 'object' and unique_count < 20:
                    value_counts = df[col].value_counts().head(5)
                    output.append("    Top values:")
                    for val, count in value_counts.items():
                        output.append(f"      {val}: {count:,} ({(count/len(df))*100:.1f}%)")
                
                # Show range for numeric columns
                elif np.issubdtype(df[col].dtype, np.number):
                    output.append(f"    Range: {df[col].min():.2f} - {df[col].max():.2f}")
                    output.append(f"    Mean: {df[col].mean():.2f} (SD: {df[col].std():.2f})")
                
                # Show date range for datetime columns
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    output.append(f"    Range: {df[col].min()} - {df[col].max()}")
                
                output.append("")
            
            # Table-specific insights
            if table_name == 'hospitalization':
                output.append("KEY METRICS:")
                output.append(f"  Unique patients: {df['patient_id'].nunique():,}")
                output.append(f"  Mortality rate: {(df['discharge_disposition'] == 'Expired').mean():.1%}")
                output.append(f"  Avg age: {df['age_at_admission'].mean():.1f} years")
                
            elif table_name == 'adt':
                output.append("LOCATION DISTRIBUTION:")
                loc_dist = df['location_category'].value_counts()
                for loc, count in loc_dist.items():
                    output.append(f"  {loc}: {count:,} ({(count/len(df))*100:.1f}%)")
                    
            elif table_name == 'respiratory_support':
                output.append("DEVICE DISTRIBUTION:")
                device_dist = df['device_category'].value_counts()
                for device, count in device_dist.items():
                    output.append(f"  {device}: {count:,} ({(count/len(df))*100:.1f}%)")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error exploring table '{table_name}': {str(e)}"
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        
        report = {
            'summary': {
                'total_tables': len(self.tables),
                'tables': self.tables
            },
            'table_quality': {}
        }
        
        for table in self.tables:
            try:
                df = pd.read_csv(self.data_path / f"{table}.csv")
                
                # Parse datetime columns
                datetime_cols = []
                for col in df.columns:
                    if 'dttm' in col or 'date' in col:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        datetime_cols.append(col)
                
                table_report = {
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'completeness': {},
                    'validity': {}
                }
                
                # Completeness metrics
                for col in df.columns:
                    completeness = (1 - df[col].isnull().mean()) * 100
                    table_report['completeness'][col] = round(completeness, 2)
                
                # Validity checks
                if table == 'hospitalization':
                    # Check for invalid discharge before admission
                    invalid_dates = df['discharge_dttm'] < df['admission_dttm']
                    table_report['validity']['valid_date_order'] = (~invalid_dates).mean() * 100
                    
                    # Check for reasonable age values
                    valid_age = (df['age_at_admission'] >= 0) & (df['age_at_admission'] <= 120)
                    table_report['validity']['valid_age_range'] = valid_age.mean() * 100
                
                elif table == 'vitals':
                    # Check for physiologically plausible values
                    if 'vital_category' in df.columns and 'vital_value' in df.columns:
                        hr_mask = df['vital_category'] == 'Heart Rate'
                        if hr_mask.any():
                            hr_values = df.loc[hr_mask, 'vital_value']
                            valid_hr = (hr_values >= 30) & (hr_values <= 250)
                            table_report['validity']['valid_heart_rate'] = valid_hr.mean() * 100
                
                report['table_quality'][table] = table_report
                
            except Exception as e:
                report['table_quality'][table] = {'error': str(e)}
        
        return report
    
    def get_linkage_report(self) -> Dict[str, Any]:
        """Check data linkage between tables"""
        
        linkage = {
            'patient_hospitalization': {},
            'hospitalization_adt': {},
            'hospitalization_clinical': {}
        }
        
        try:
            # Load key tables
            patients = pd.read_csv(self.data_path / "patient.csv")
            hospitalizations = pd.read_csv(self.data_path / "hospitalization.csv")
            
            # Check patient-hospitalization linkage
            hosp_patients = hospitalizations['patient_id'].unique()
            all_patients = patients['patient_id'].unique()
            
            linkage['patient_hospitalization'] = {
                'hospitalizations_with_valid_patient': 
                    (hospitalizations['patient_id'].isin(all_patients)).mean() * 100,
                'patients_with_hospitalizations': 
                    (patients['patient_id'].isin(hosp_patients)).mean() * 100
            }
            
            # Check hospitalization-ADT linkage
            if 'adt' in self.tables:
                adt = pd.read_csv(self.data_path / "adt.csv")
                hosp_ids = hospitalizations['hospitalization_id'].unique()
                
                linkage['hospitalization_adt'] = {
                    'adt_with_valid_hospitalization':
                        (adt['hospitalization_id'].isin(hosp_ids)).mean() * 100,
                    'hospitalizations_with_adt':
                        (hospitalizations['hospitalization_id'].isin(
                            adt['hospitalization_id'].unique()
                        )).mean() * 100
                }
            
        except Exception as e:
            linkage['error'] = str(e)
        
        return linkage