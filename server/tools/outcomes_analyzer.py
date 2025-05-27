"""
Outcomes Analysis Tool for CLIF MCP Server
Focuses on mortality, length of stay, complications, and comparative effectiveness
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import mcnemar
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class OutcomesAnalyzer:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.tables = self._load_tables()
        
    def _load_tables(self) -> Dict[str, pd.DataFrame]:
        """Load required CLIF tables"""
        tables = {}
        required = ['patient', 'hospitalization', 'adt', 'respiratory_support', 
                   'medication_administration', 'patient_assessments', 'labs']
        
        for table in required:
            filepath = self.data_path / f"{table}.csv"
            if filepath.exists():
                tables[table] = pd.read_csv(filepath, parse_dates=True)
        
        return tables
    
    def analyze_outcomes(self, cohort_id: Optional[str], outcomes: List[str],
                        stratify_by: List[str], adjust_for: List[str]) -> Dict[str, Any]:
        """Main outcomes analysis function"""
        
        # Load cohort or use full dataset
        if cohort_id:
            cohort = self._load_cohort(cohort_id)
        else:
            cohort = self.tables['hospitalization']
        
        results = {
            'cohort_size': len(cohort),
            'outcomes': {},
            'stratified': {},
            'adjusted': {}
        }
        
        # Analyze each outcome
        for outcome in outcomes:
            if outcome == 'mortality':
                results['outcomes'][outcome] = self._analyze_mortality(cohort)
            elif outcome == 'icu_mortality':
                results['outcomes'][outcome] = self._analyze_icu_mortality(cohort)
            elif outcome == 'hospital_mortality':
                results['outcomes'][outcome] = self._analyze_hospital_mortality(cohort)
            elif outcome == 'icu_los':
                results['outcomes'][outcome] = self._analyze_icu_los(cohort)
            elif outcome == 'hospital_los':
                results['outcomes'][outcome] = self._analyze_hospital_los(cohort)
            elif outcome == 'ventilator_days':
                results['outcomes'][outcome] = self._analyze_ventilator_days(cohort)
            elif outcome == 'readmission_30d':
                results['outcomes'][outcome] = self._analyze_readmission(cohort)
            elif outcome == 'aki':
                results['outcomes'][outcome] = self._analyze_aki(cohort)
            elif outcome == 'delirium':
                results['outcomes'][outcome] = self._analyze_delirium(cohort)
        
        # Stratified analysis
        if stratify_by:
            results['stratified'] = self._stratified_analysis(cohort, outcomes, stratify_by)
        
        # Adjusted analysis
        if adjust_for:
            results['adjusted'] = self._adjusted_analysis(cohort, outcomes, adjust_for)
        
        return results
    
    def _analyze_mortality(self, cohort: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall mortality"""
        mortality = cohort['discharge_disposition'] == 'Expired'
        
        return {
            'rate': float(mortality.mean()),
            'count': int(mortality.sum()),
            'total': len(cohort),
            'ci_95': self._proportion_ci(mortality.sum(), len(cohort))
        }
    
    def _analyze_icu_mortality(self, cohort: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ICU mortality"""
        # Get ICU stays
        icu_stays = self._get_icu_stays(cohort)
        
        # Determine if death occurred during ICU stay
        icu_mortality = []
        for _, hosp in cohort.iterrows():
            if hosp['discharge_disposition'] == 'Expired':
                # Check if death occurred during ICU
                icu = icu_stays[icu_stays['hospitalization_id'] == hosp['hospitalization_id']]
                if not icu.empty:
                    last_icu = icu.iloc[-1]
                    if pd.to_datetime(hosp['discharge_dttm']) <= pd.to_datetime(last_icu['out_dttm']):
                        icu_mortality.append(True)
                    else:
                        icu_mortality.append(False)
                else:
                    icu_mortality.append(False)
            else:
                icu_mortality.append(False)
        
        icu_mortality = pd.Series(icu_mortality)
        
        return {
            'rate': float(icu_mortality.mean()),
            'count': int(icu_mortality.sum()),
            'total': len(cohort),
            'ci_95': self._proportion_ci(icu_mortality.sum(), len(cohort))
        }
    
    def _analyze_hospital_mortality(self, cohort: pd.DataFrame) -> Dict[str, Any]:
        """Same as overall mortality but explicit"""
        return self._analyze_mortality(cohort)
    
    def _analyze_icu_los(self, cohort: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ICU length of stay"""
        icu_stays = self._get_icu_stays(cohort)
        
        # Calculate total ICU hours per hospitalization
        icu_los = []
        for hosp_id in cohort['hospitalization_id']:
            stays = icu_stays[icu_stays['hospitalization_id'] == hosp_id]
            if not stays.empty:
                total_hours = 0
                for _, stay in stays.iterrows():
                    duration = pd.to_datetime(stay['out_dttm']) - pd.to_datetime(stay['in_dttm'])
                    total_hours += duration.total_seconds() / 3600
                icu_los.append(total_hours / 24)  # Convert to days
            else:
                icu_los.append(0)
        
        icu_los = pd.Series(icu_los)
        icu_los = icu_los[icu_los > 0]  # Only include patients with ICU stay
        
        return {
            'mean': float(icu_los.mean()),
            'median': float(icu_los.median()),
            'q1': float(icu_los.quantile(0.25)),
            'q3': float(icu_los.quantile(0.75)),
            'min': float(icu_los.min()),
            'max': float(icu_los.max()),
            'n': len(icu_los)
        }
    
    def _analyze_hospital_los(self, cohort: pd.DataFrame) -> Dict[str, Any]:
        """Analyze hospital length of stay"""
        los = []
        for _, hosp in cohort.iterrows():
            duration = pd.to_datetime(hosp['discharge_dttm']) - pd.to_datetime(hosp['admission_dttm'])
            los.append(duration.total_seconds() / 86400)  # Days
        
        los = pd.Series(los)
        
        return {
            'mean': float(los.mean()),
            'median': float(los.median()),
            'q1': float(los.quantile(0.25)),
            'q3': float(los.quantile(0.75)),
            'min': float(los.min()),
            'max': float(los.max()),
            'n': len(los)
        }
    
    def _analyze_ventilator_days(self, cohort: pd.DataFrame) -> Dict[str, Any]:
        """Analyze mechanical ventilation duration"""
        resp_support = self.tables['respiratory_support']
        
        vent_days = []
        for hosp_id in cohort['hospitalization_id']:
            vent = resp_support[
                (resp_support['hospitalization_id'] == hosp_id) &
                (resp_support['device_category'] == 'Invasive Mechanical Ventilation')
            ]
            
            if not vent.empty:
                total_hours = 0
                for _, period in vent.iterrows():
                    duration = pd.to_datetime(period['stop_dttm']) - pd.to_datetime(period['start_dttm'])
                    total_hours += duration.total_seconds() / 3600
                vent_days.append(total_hours / 24)
            else:
                vent_days.append(0)
        
        vent_days = pd.Series(vent_days)
        
        return {
            'mean_all': float(vent_days.mean()),
            'mean_ventilated': float(vent_days[vent_days > 0].mean()) if any(vent_days > 0) else 0,
            'percent_ventilated': float((vent_days > 0).mean()),
            'median_ventilated': float(vent_days[vent_days > 0].median()) if any(vent_days > 0) else 0,
            'n_ventilated': int((vent_days > 0).sum())
        }
    
    def _analyze_readmission(self, cohort: pd.DataFrame) -> Dict[str, Any]:
        """Analyze 30-day readmission"""
        readmissions = 0
        eligible = 0
        
        # Group by patient
        for patient_id, patient_hosps in cohort.groupby('patient_id'):
            patient_hosps = patient_hosps.sort_values('admission_dttm')
            
            for i in range(len(patient_hosps) - 1):
                current = patient_hosps.iloc[i]
                next_hosp = patient_hosps.iloc[i + 1]
                
                # Skip if patient died
                if current['discharge_disposition'] == 'Expired':
                    continue
                    
                eligible += 1
                
                # Check if readmission within 30 days
                days_between = (pd.to_datetime(next_hosp['admission_dttm']) - 
                              pd.to_datetime(current['discharge_dttm'])).days
                
                if days_between <= 30:
                    readmissions += 1
        
        return {
            'rate': float(readmissions / eligible) if eligible > 0 else 0,
            'count': readmissions,
            'eligible': eligible,
            'ci_95': self._proportion_ci(readmissions, eligible) if eligible > 0 else (0, 0)
        }
    
    def _analyze_aki(self, cohort: pd.DataFrame) -> Dict[str, Any]:
        """Analyze acute kidney injury (simplified - creatinine increase)"""
        labs = self.tables['labs']
        
        aki_count = 0
        
        for hosp_id in cohort['hospitalization_id']:
            creat = labs[
                (labs['hospitalization_id'] == hosp_id) &
                (labs['lab_category'] == 'Creatinine')
            ].sort_values('lab_result_dttm')
            
            if len(creat) >= 2:
                # Simple definition: 50% increase from baseline
                baseline = creat.iloc[0]['lab_value']
                max_creat = creat['lab_value'].max()
                
                if max_creat >= baseline * 1.5:
                    aki_count += 1
        
        return {
            'rate': float(aki_count / len(cohort)),
            'count': aki_count,
            'total': len(cohort),
            'ci_95': self._proportion_ci(aki_count, len(cohort))
        }
    
    def _analyze_delirium(self, cohort: pd.DataFrame) -> Dict[str, Any]:
        """Analyze delirium (CAM-ICU positive)"""
        assessments = self.tables['patient_assessments']
        
        delirium_count = 0
        
        for hosp_id in cohort['hospitalization_id']:
            cam = assessments[
                (assessments['hospitalization_id'] == hosp_id) &
                (assessments['assessment_category'] == 'CAM-ICU')
            ]
            
            if not cam.empty and any(cam['assessment_value'] == 1):
                delirium_count += 1
        
        return {
            'rate': float(delirium_count / len(cohort)),
            'count': delirium_count,
            'total': len(cohort),
            'ci_95': self._proportion_ci(delirium_count, len(cohort))
        }
    
    def _get_icu_stays(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Get ICU stays for cohort"""
        adt = self.tables['adt']
        hosp_ids = cohort['hospitalization_id'].tolist()
        
        return adt[
            (adt['hospitalization_id'].isin(hosp_ids)) &
            (adt['location_category'] == 'ICU')
        ]
    
    def _proportion_ci(self, successes: int, total: int, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate confidence interval for proportion"""
        from statsmodels.stats.proportion import proportion_confint
        
        if total == 0:
            return (0, 0)
        
        ci_low, ci_high = proportion_confint(successes, total, alpha=alpha, method='wilson')
        return (float(ci_low), float(ci_high))
    
    def _stratified_analysis(self, cohort: pd.DataFrame, outcomes: List[str], 
                           stratify_by: List[str]) -> Dict[str, Any]:
        """Perform stratified analysis"""
        results = {}
        
        for var in stratify_by:
            if var in cohort.columns:
                results[var] = {}
                
                for group, group_data in cohort.groupby(var):
                    results[var][str(group)] = {}
                    
                    for outcome in outcomes:
                        if outcome == 'mortality':
                            results[var][str(group)][outcome] = self._analyze_mortality(group_data)
                        elif outcome == 'icu_los':
                            results[var][str(group)][outcome] = self._analyze_icu_los(group_data)
                        # Add other outcomes as needed
        
        return results
    
    def _adjusted_analysis(self, cohort: pd.DataFrame, outcomes: List[str],
                         adjust_for: List[str]) -> Dict[str, Any]:
        """Perform adjusted analysis using regression"""
        results = {}
        
        # Prepare covariates
        X = cohort[adjust_for].copy()
        X = pd.get_dummies(X, drop_first=True)
        
        # Standardize continuous variables
        scaler = StandardScaler()
        continuous_cols = X.select_dtypes(include=[np.number]).columns
        X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
        
        for outcome in outcomes:
            if outcome in ['mortality', 'hospital_mortality']:
                # Logistic regression for binary outcomes
                y = (cohort['discharge_disposition'] == 'Expired').astype(int)
                
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)
                
                results[outcome] = {
                    'coefficients': dict(zip(X.columns, model.coef_[0])),
                    'intercept': float(model.intercept_[0]),
                    'score': float(model.score(X, y))
                }
        
        return results
    
    def compare_cohorts(self, cohort1_id: str, cohort2_id: str,
                       outcomes: List[str], adjustment_method: str) -> Dict[str, Any]:
        """Compare outcomes between two cohorts"""
        cohort1 = self._load_cohort(cohort1_id)
        cohort2 = self._load_cohort(cohort2_id)
        
        results = {
            'cohort1_size': len(cohort1),
            'cohort2_size': len(cohort2),
            'comparisons': {}
        }
        
        for outcome in outcomes:
            if outcome in ['mortality', 'hospital_mortality']:
                results['comparisons'][outcome] = self._compare_binary_outcome(
                    cohort1, cohort2, outcome, adjustment_method
                )
            elif outcome in ['icu_los', 'hospital_los']:
                results['comparisons'][outcome] = self._compare_continuous_outcome(
                    cohort1, cohort2, outcome, adjustment_method
                )
        
        return results
    
    def _compare_binary_outcome(self, cohort1: pd.DataFrame, cohort2: pd.DataFrame,
                              outcome: str, adjustment_method: str) -> Dict[str, Any]:
        """Compare binary outcomes between cohorts"""
        # Get outcome for each cohort
        if outcome in ['mortality', 'hospital_mortality']:
            outcome1 = (cohort1['discharge_disposition'] == 'Expired')
            outcome2 = (cohort2['discharge_disposition'] == 'Expired')
        
        n1, n2 = len(cohort1), len(cohort2)
        x1, x2 = outcome1.sum(), outcome2.sum()
        
        # Unadjusted comparison
        stat, pval = proportions_ztest([x1, x2], [n1, n2])
        
        risk_diff = outcome1.mean() - outcome2.mean()
        risk_ratio = outcome1.mean() / outcome2.mean() if outcome2.mean() > 0 else np.inf
        
        result = {
            'cohort1_rate': float(outcome1.mean()),
            'cohort2_rate': float(outcome2.mean()),
            'risk_difference': float(risk_diff),
            'risk_ratio': float(risk_ratio),
            'p_value': float(pval),
            'test': 'z-test for proportions'
        }
        
        # Add confidence intervals
        result['risk_diff_ci'] = self._risk_difference_ci(x1, n1, x2, n2)
        
        return result
    
    def _compare_continuous_outcome(self, cohort1: pd.DataFrame, cohort2: pd.DataFrame,
                                  outcome: str, adjustment_method: str) -> Dict[str, Any]:
        """Compare continuous outcomes between cohorts"""
        if outcome == 'icu_los':
            values1 = self._get_icu_los_values(cohort1)
            values2 = self._get_icu_los_values(cohort2)
        elif outcome == 'hospital_los':
            values1 = self._get_hospital_los_values(cohort1)
            values2 = self._get_hospital_los_values(cohort2)
        
        # T-test for continuous outcomes
        stat, pval = stats.ttest_ind(values1, values2)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pval = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        
        return {
            'cohort1_mean': float(values1.mean()),
            'cohort1_median': float(values1.median()),
            'cohort2_mean': float(values2.mean()),
            'cohort2_median': float(values2.median()),
            'mean_difference': float(values1.mean() - values2.mean()),
            't_test_p': float(pval),
            'mann_whitney_p': float(u_pval)
        }
    
    def _get_icu_los_values(self, cohort: pd.DataFrame) -> pd.Series:
        """Extract ICU LOS values for cohort"""
        icu_stays = self._get_icu_stays(cohort)
        
        los_values = []
        for hosp_id in cohort['hospitalization_id']:
            stays = icu_stays[icu_stays['hospitalization_id'] == hosp_id]
            if not stays.empty:
                total_hours = 0
                for _, stay in stays.iterrows():
                    duration = pd.to_datetime(stay['out_dttm']) - pd.to_datetime(stay['in_dttm'])
                    total_hours += duration.total_seconds() / 3600
                los_values.append(total_hours / 24)
        
        return pd.Series(los_values)
    
    def _get_hospital_los_values(self, cohort: pd.DataFrame) -> pd.Series:
        """Extract hospital LOS values for cohort"""
        los_values = []
        for _, hosp in cohort.iterrows():
            duration = pd.to_datetime(hosp['discharge_dttm']) - pd.to_datetime(hosp['admission_dttm'])
            los_values.append(duration.total_seconds() / 86400)
        
        return pd.Series(los_values)
    
    def _risk_difference_ci(self, x1: int, n1: int, x2: int, n2: int, 
                           alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate confidence interval for risk difference"""
        p1, p2 = x1 / n1, x2 / n2
        rd = p1 - p2
        
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        z = stats.norm.ppf(1 - alpha / 2)
        
        ci_low = rd - z * se
        ci_high = rd + z * se
        
        return (float(ci_low), float(ci_high))
    
    def _load_cohort(self, cohort_id: str) -> pd.DataFrame:
        """Load saved cohort"""
        cohort_path = self.data_path.parent / "cohorts" / f"{cohort_id}.csv"
        if cohort_path.exists():
            return pd.read_csv(cohort_path)
        else:
            # Return full hospitalization table if cohort not found
            return self.tables['hospitalization']
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format analysis results for display"""
        output = []
        output.append(f"Cohort Analysis Results")
        output.append(f"{'=' * 50}")
        output.append(f"Cohort size: {results['cohort_size']} hospitalizations\n")
        
        # Format each outcome
        for outcome, data in results['outcomes'].items():
            output.append(f"{outcome.upper().replace('_', ' ')}:")
            
            if 'rate' in data:
                output.append(f"  Rate: {data['rate']:.1%} ({data['count']}/{data['total']})")
                output.append(f"  95% CI: ({data['ci_95'][0]:.1%}, {data['ci_95'][1]:.1%})")
            elif 'mean' in data:
                output.append(f"  Mean: {data['mean']:.1f} days")
                output.append(f"  Median (IQR): {data['median']:.1f} ({data['q1']:.1f}-{data['q3']:.1f})")
                output.append(f"  Range: {data['min']:.1f}-{data['max']:.1f}")
            
            output.append("")
        
        return "\n".join(output)
    
    def format_comparison(self, results: Dict[str, Any]) -> str:
        """Format comparison results for display"""
        output = []
        output.append(f"Cohort Comparison Results")
        output.append(f"{'=' * 50}")
        output.append(f"Cohort 1: n={results['cohort1_size']}")
        output.append(f"Cohort 2: n={results['cohort2_size']}\n")
        
        for outcome, comp in results['comparisons'].items():
            output.append(f"{outcome.upper().replace('_', ' ')}:")
            
            if 'risk_difference' in comp:
                output.append(f"  Cohort 1: {comp['cohort1_rate']:.1%}")
                output.append(f"  Cohort 2: {comp['cohort2_rate']:.1%}")
                output.append(f"  Risk Difference: {comp['risk_difference']:.1%}")
                output.append(f"  Risk Ratio: {comp['risk_ratio']:.2f}")
                output.append(f"  P-value: {comp['p_value']:.4f}")
            elif 'mean_difference' in comp:
                output.append(f"  Cohort 1: {comp['cohort1_mean']:.1f} days (median {comp['cohort1_median']:.1f})")
                output.append(f"  Cohort 2: {comp['cohort2_mean']:.1f} days (median {comp['cohort2_median']:.1f})")
                output.append(f"  Mean Difference: {comp['mean_difference']:.1f} days")
                output.append(f"  T-test p-value: {comp['t_test_p']:.4f}")
                output.append(f"  Mann-Whitney p-value: {comp['mann_whitney_p']:.4f}")
            
            output.append("")
        
        return "\n".join(output)