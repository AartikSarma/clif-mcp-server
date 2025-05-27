"""
Synthetic CLIF Dataset Generator
Generates realistic ICU data following CLIF 2.1.0 schema
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import os
from typing import Dict, List, Tuple

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)


class CLIFSyntheticGenerator:
    def __init__(self, n_patients: int = 1000, output_dir: str = "./data/synthetic"):
        self.n_patients = n_patients
        self.output_dir = output_dir
        self.patient_data = {}
        self.hospitalization_data = []
        self.adt_data = []
        self.vitals_data = []
        self.labs_data = []
        self.respiratory_support_data = []
        self.medication_admin_data = []
        self.patient_assessments_data = []
        self.position_data = []
        
    def generate_all_tables(self):
        """Generate all CLIF beta tables"""
        print("Generating synthetic CLIF dataset...")
        
        # Core tables
        self._generate_patients()
        self._generate_hospitalizations()
        self._generate_adt()
        
        # Clinical data tables
        self._generate_vitals()
        self._generate_labs()
        self._generate_respiratory_support()
        self._generate_medications()
        self._generate_assessments()
        self._generate_positions()
        
        # Save all tables
        self._save_all_tables()
        print(f"Synthetic dataset generated in {self.output_dir}")
    
    def _generate_patients(self):
        """Generate patient demographics"""
        races = ['White', 'Black or African American', 'Asian', 'Other', 'Unknown']
        ethnicities = ['Not Hispanic or Latino', 'Hispanic or Latino', 'Unknown']
        sexes = ['Male', 'Female']
        
        for i in range(self.n_patients):
            patient_id = f"PT{i+1:06d}"
            birth_date = fake.date_of_birth(minimum_age=18, maximum_age=95)
            
            self.patient_data[patient_id] = {
                'patient_id': patient_id,
                'race': np.random.choice(races, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
                'ethnicity': np.random.choice(ethnicities, p=[0.7, 0.25, 0.05]),
                'sex': np.random.choice(sexes, p=[0.55, 0.45]),
                'date_of_birth': birth_date
            }
    
    def _generate_hospitalizations(self):
        """Generate hospital admissions with varying ICU stays"""
        admission_sources = ['Emergency Department', 'Direct Admission', 
                           'Transfer from Another Hospital', 'Operating Room']
        discharge_dispositions = ['Home', 'Skilled Nursing Facility', 
                                'Expired', 'Hospice', 'Transfer to Another Hospital']
        
        hosp_id = 1
        for patient_id in self.patient_data.keys():
            # Each patient has 1-3 hospitalizations
            n_hosps = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
            
            admission_date = datetime.now() - timedelta(days=random.randint(30, 365))
            
            for _ in range(n_hosps):
                # ICU length of stay: 1-30 days, skewed towards shorter stays
                icu_los = int(np.random.gamma(2, 2))
                icu_los = max(1, min(icu_los, 30))
                
                # Total hospital LOS is ICU + additional days
                total_los = icu_los + random.randint(0, 7)
                
                discharge_date = admission_date + timedelta(days=total_los)
                
                # Mortality correlates with longer ICU stays
                mortality_prob = min(0.05 + (icu_los / 100), 0.3)
                is_mortality = np.random.random() < mortality_prob
                
                hosp = {
                    'hospitalization_id': f"HOSP{hosp_id:06d}",
                    'patient_id': patient_id,
                    'admission_dttm': admission_date,
                    'discharge_dttm': discharge_date,
                    'admission_source': np.random.choice(admission_sources),
                    'discharge_disposition': 'Expired' if is_mortality else np.random.choice(
                        [d for d in discharge_dispositions if d != 'Expired']
                    ),
                    'age_at_admission': (admission_date.date() - 
                                       self.patient_data[patient_id]['date_of_birth']).days // 365
                }
                
                self.hospitalization_data.append(hosp)
                hosp_id += 1
                
                # Next admission at least 30 days later
                admission_date = discharge_date + timedelta(days=random.randint(30, 180))
    
    def _generate_adt(self):
        """Generate ADT (Admission/Discharge/Transfer) events"""
        location_categories = ['Emergency Department', 'ICU', 'Ward', 'Operating Room']
        
        for hosp in self.hospitalization_data:
            hosp_id = hosp['hospitalization_id']
            admission = hosp['admission_dttm']
            discharge = hosp['discharge_dttm']
            
            current_time = admission
            
            # Start in ED or direct to ICU
            if hosp['admission_source'] == 'Emergency Department':
                # ED stay 2-8 hours
                ed_duration = timedelta(hours=random.randint(2, 8))
                self.adt_data.append({
                    'hospitalization_id': hosp_id,
                    'in_dttm': current_time,
                    'out_dttm': current_time + ed_duration,
                    'location_category': 'Emergency Department'
                })
                current_time += ed_duration
            
            # ICU stay
            icu_start = current_time
            # Calculate ICU duration based on total stay
            total_hours = (discharge - admission).total_seconds() / 3600
            icu_hours = total_hours * random.uniform(0.3, 0.8)  # ICU is 30-80% of stay
            icu_end = icu_start + timedelta(hours=icu_hours)
            
            self.adt_data.append({
                'hospitalization_id': hosp_id,
                'in_dttm': icu_start,
                'out_dttm': icu_end,
                'location_category': 'ICU'
            })
            
            # Potential ward stay after ICU
            if icu_end < discharge:
                self.adt_data.append({
                    'hospitalization_id': hosp_id,
                    'in_dttm': icu_end,
                    'out_dttm': discharge,
                    'location_category': 'Ward'
                })
    
    def _generate_vitals(self):
        """Generate vital signs data"""
        for adt in self.adt_data:
            if adt['location_category'] != 'ICU':
                continue
                
            hosp_id = adt['hospitalization_id']
            start = adt['in_dttm']
            end = adt['out_dttm']
            
            # Vitals every 1-4 hours in ICU
            current = start
            while current < end:
                # Generate correlated vital signs
                is_unstable = random.random() < 0.2
                
                hr = np.random.normal(90 if is_unstable else 75, 15)
                bp_sys = np.random.normal(110 if is_unstable else 120, 20)
                bp_dia = bp_sys * random.uniform(0.55, 0.65)
                temp = np.random.normal(37.5 if is_unstable else 37.0, 0.5)
                rr = np.random.normal(22 if is_unstable else 16, 4)
                spo2 = np.random.normal(94 if is_unstable else 97, 3)
                
                vitals = {
                    'hospitalization_id': hosp_id,
                    'recorded_dttm': current,
                    'vital_category': 'Heart Rate',
                    'vital_value': max(40, min(180, hr))
                }
                self.vitals_data.append(vitals)
                
                self.vitals_data.append({
                    'hospitalization_id': hosp_id,
                    'recorded_dttm': current,
                    'vital_category': 'Systolic Blood Pressure',
                    'vital_value': max(60, min(200, bp_sys))
                })
                
                self.vitals_data.append({
                    'hospitalization_id': hosp_id,
                    'recorded_dttm': current,
                    'vital_category': 'Temperature',
                    'vital_value': max(35, min(40, temp))
                })
                
                self.vitals_data.append({
                    'hospitalization_id': hosp_id,
                    'recorded_dttm': current,
                    'vital_category': 'SpO2',
                    'vital_value': max(85, min(100, spo2))
                })
                
                current += timedelta(hours=random.randint(1, 4))
    
    def _generate_labs(self):
        """Generate laboratory results"""
        common_labs = [
            ('Hemoglobin', 8, 16, 'g/dL'),
            ('White Blood Cell Count', 4, 20, 'K/uL'),
            ('Platelet Count', 50, 400, 'K/uL'),
            ('Creatinine', 0.5, 3.0, 'mg/dL'),
            ('Glucose', 60, 300, 'mg/dL'),
            ('Sodium', 130, 150, 'mEq/L'),
            ('Potassium', 3.0, 5.5, 'mEq/L'),
            ('Lactate', 0.5, 5.0, 'mmol/L')
        ]
        
        for adt in self.adt_data:
            if adt['location_category'] != 'ICU':
                continue
                
            hosp_id = adt['hospitalization_id']
            start = adt['in_dttm']
            end = adt['out_dttm']
            
            # Labs every 6-24 hours
            current = start
            while current < end:
                for lab_name, low, high, unit in common_labs:
                    if random.random() < 0.8:  # Not all labs ordered every time
                        value = np.random.uniform(low, high)
                        
                        self.labs_data.append({
                            'hospitalization_id': hosp_id,
                            'lab_order_dttm': current,
                            'lab_result_dttm': current + timedelta(hours=1),
                            'lab_category': lab_name,
                            'lab_value': round(value, 2),
                            'lab_unit': unit
                        })
                
                current += timedelta(hours=random.randint(6, 24))
    
    def _generate_respiratory_support(self):
        """Generate respiratory support device data"""
        devices = [
            ('None', 0.4),
            ('Nasal Cannula', 0.2),
            ('High Flow Nasal Cannula', 0.15),
            ('Non-Invasive Ventilation', 0.1),
            ('Invasive Mechanical Ventilation', 0.15)
        ]
        
        for adt in self.adt_data:
            if adt['location_category'] != 'ICU':
                continue
                
            hosp_id = adt['hospitalization_id']
            start = adt['in_dttm']
            end = adt['out_dttm']
            
            current = start
            while current < end:
                device = np.random.choice([d[0] for d in devices], 
                                        p=[d[1] for d in devices])
                
                duration = timedelta(hours=random.randint(4, 48))
                device_end = min(current + duration, end)
                
                if device != 'None':
                    self.respiratory_support_data.append({
                        'hospitalization_id': hosp_id,
                        'start_dttm': current,
                        'stop_dttm': device_end,
                        'device_category': device,
                        'fio2': random.randint(21, 100) if device != 'None' else 21
                    })
                
                current = device_end
    
    def _generate_medications(self):
        """Generate medication administration data"""
        common_meds = [
            ('Propofol', 'Sedative', 'Continuous'),
            ('Fentanyl', 'Analgesic', 'Continuous'),
            ('Norepinephrine', 'Vasopressor', 'Continuous'),
            ('Vancomycin', 'Antibiotic', 'Intermittent'),
            ('Piperacillin-Tazobactam', 'Antibiotic', 'Intermittent'),
            ('Heparin', 'Anticoagulant', 'Continuous'),
            ('Insulin', 'Hormone', 'Continuous'),
            ('Furosemide', 'Diuretic', 'Intermittent')
        ]
        
        for adt in self.adt_data:
            if adt['location_category'] != 'ICU':
                continue
                
            hosp_id = adt['hospitalization_id']
            start = adt['in_dttm']
            end = adt['out_dttm']
            
            # Select random medications for this stay
            n_meds = random.randint(3, 7)
            selected_meds = random.sample(common_meds, n_meds)
            
            for med_name, med_class, admin_type in selected_meds:
                if admin_type == 'Continuous':
                    # Continuous infusion
                    med_start = start + timedelta(hours=random.randint(0, 12))
                    remaining_hours = int((end - med_start).total_seconds() / 3600)
                    if remaining_hours > 24:
                        duration_hours = random.randint(24, remaining_hours)
                    else:
                        duration_hours = max(1, remaining_hours)
                    med_stop = med_start + timedelta(hours=duration_hours)
                    
                    self.medication_admin_data.append({
                        'hospitalization_id': hosp_id,
                        'admin_dttm': med_start,
                        'medication_name': med_name,
                        'medication_category': med_class,
                        'route': 'Intravenous',
                        'dose': random.randint(1, 20),
                        'dose_unit': 'mcg/kg/min' if med_class == 'Vasopressor' else 'mg/hr'
                    })
                else:
                    # Intermittent dosing
                    current = start
                    interval = random.choice([6, 8, 12, 24])
                    
                    while current < end:
                        self.medication_admin_data.append({
                            'hospitalization_id': hosp_id,
                            'admin_dttm': current,
                            'medication_name': med_name,
                            'medication_category': med_class,
                            'route': 'Intravenous',
                            'dose': random.randint(250, 1000),
                            'dose_unit': 'mg'
                        })
                        current += timedelta(hours=interval)
    
    def _generate_assessments(self):
        """Generate patient assessment scores"""
        for adt in self.adt_data:
            if adt['location_category'] != 'ICU':
                continue
                
            hosp_id = adt['hospitalization_id']
            start = adt['in_dttm']
            end = adt['out_dttm']
            
            # Daily assessments
            current = start
            while current < end:
                # RASS (Richmond Agitation-Sedation Scale)
                self.patient_assessments_data.append({
                    'hospitalization_id': hosp_id,
                    'assessment_dttm': current,
                    'assessment_category': 'RASS',
                    'assessment_value': random.randint(-5, 4)
                })
                
                # CAM-ICU (Confusion Assessment Method)
                self.patient_assessments_data.append({
                    'hospitalization_id': hosp_id,
                    'assessment_dttm': current,
                    'assessment_category': 'CAM-ICU',
                    'assessment_value': random.choice([0, 1])  # 0=negative, 1=positive
                })
                
                # Pain score
                self.patient_assessments_data.append({
                    'hospitalization_id': hosp_id,
                    'assessment_dttm': current,
                    'assessment_category': 'Pain Score',
                    'assessment_value': random.randint(0, 10)
                })
                
                current += timedelta(hours=24)
    
    def _generate_positions(self):
        """Generate patient position data"""
        positions = ['Supine', 'Semi-Fowler', 'Prone', 'Left Lateral', 'Right Lateral']
        
        for adt in self.adt_data:
            if adt['location_category'] != 'ICU':
                continue
                
            hosp_id = adt['hospitalization_id']
            start = adt['in_dttm']
            end = adt['out_dttm']
            
            # Position changes every 2-4 hours
            current = start
            while current < end:
                duration = timedelta(hours=random.randint(2, 4))
                position_end = min(current + duration, end)
                
                self.position_data.append({
                    'hospitalization_id': hosp_id,
                    'start_dttm': current,
                    'stop_dttm': position_end,
                    'position': random.choice(positions)
                })
                
                current = position_end
    
    def _save_all_tables(self):
        """Save all generated tables as CSV files"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Convert and save each table
        tables = {
            'patient': pd.DataFrame(list(self.patient_data.values())),
            'hospitalization': pd.DataFrame(self.hospitalization_data),
            'adt': pd.DataFrame(self.adt_data),
            'vitals': pd.DataFrame(self.vitals_data),
            'labs': pd.DataFrame(self.labs_data),
            'respiratory_support': pd.DataFrame(self.respiratory_support_data),
            'medication_administration': pd.DataFrame(self.medication_admin_data),
            'patient_assessments': pd.DataFrame(self.patient_assessments_data),
            'position': pd.DataFrame(self.position_data)
        }
        
        for name, df in tables.items():
            filepath = os.path.join(self.output_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            print(f"  - {name}.csv: {len(df)} records")


def main():
    """Main entry point for synthetic data generation"""
    generator = CLIFSyntheticGenerator(n_patients=1000)
    generator.generate_all_tables()


if __name__ == "__main__":
    main()