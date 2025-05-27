"""
Privacy Guard for CLIF MCP Server
Ensures data privacy and prevents leakage of patient information
"""

import re
from typing import Any, Dict, List, Union, Tuple
import hashlib
import json
import pandas as pd


class PrivacyGuard:
    """Enforce privacy constraints on all outputs"""
    
    def __init__(self):
        self.min_cell_size = 10  # Minimum cell size for reporting
        self.suppression_threshold = 5  # Suppress if less than this
        self.audit_log = []
        
    def sanitize_output(self, output: Any) -> Any:
        """Sanitize any output to ensure privacy"""
        
        if isinstance(output, str):
            return self._sanitize_string(output)
        elif isinstance(output, dict):
            return self._sanitize_dict(output)
        elif isinstance(output, list):
            return [self.sanitize_output(item) for item in output]
        elif isinstance(output, (int, float)):
            return self._sanitize_number(output)
        else:
            return output
    
    def _sanitize_string(self, text: str) -> str:
        """Remove potential PHI from text"""
        
        # Remove potential patient IDs (various formats)
        text = re.sub(r'\b(PT|PAT|MRN|ID)[0-9]{4,}\b', '[PATIENT_ID]', text)
        
        # Remove dates that could be identifying
        text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]', text)
        text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]', text)
        
        # Remove potential names (basic heuristic)
        text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[NAME]', text)
        
        # Remove specific ages over 89 (HIPAA identifier)
        text = re.sub(r'\b(9[0-9]|1[0-9]{2})\s*years?\s*old\b', '>89 years old', text)
        
        return text
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary outputs"""
        
        sanitized = {}
        
        for key, value in data.items():
            # Skip patient-level identifiers
            if key in ['patient_id', 'hospitalization_id', 'mrn']:
                continue
            
            # Apply cell suppression for small counts
            if key in ['count', 'n', 'total'] and isinstance(value, (int, float)):
                if value < self.suppression_threshold:
                    sanitized[key] = f"<{self.suppression_threshold}"
                else:
                    sanitized[key] = value
            else:
                sanitized[key] = self.sanitize_output(value)
        
        return sanitized
    
    def _sanitize_number(self, value: Union[int, float]) -> Union[int, float, str]:
        """Sanitize numeric values"""
        
        # For very small counts, suppress
        if isinstance(value, int) and 0 < value < self.suppression_threshold:
            return f"<{self.suppression_threshold}"
        
        # Round percentages to avoid re-identification
        if isinstance(value, float) and 0 <= value <= 1:
            return round(value, 3)  # Limit precision
        
        return value
    
    def check_query_safety(self, query: Dict[str, Any]) -> bool:
        """Check if a query is safe to execute"""
        
        # Log the query
        self.audit_log.append({
            'timestamp': str(pd.Timestamp.now()),
            'query': query,
            'action': 'query_check'
        })
        
        # Check for attempts to extract individual records
        if 'limit' in query and query['limit'] < self.min_cell_size:
            return False
        
        # Check for overly specific criteria that could identify individuals
        criteria_count = 0
        if 'criteria' in query:
            for key, value in query['criteria'].items():
                if value is not None:
                    criteria_count += 1
        
        # Too many specific criteria could identify individuals
        if criteria_count > 5:
            return False
        
        return True
    
    def apply_differential_privacy(self, value: float, epsilon: float = 1.0) -> float:
        """Apply differential privacy noise to numeric values"""
        import numpy as np
        
        # Add Laplace noise
        sensitivity = 1.0  # Assuming normalized data
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        
        return value + noise
    
    def generate_safe_summary(self, data: pd.DataFrame, 
                            group_vars: List[str],
                            summary_vars: List[str]) -> pd.DataFrame:
        """Generate privacy-preserving summary statistics"""
        
        # Group data
        grouped = data.groupby(group_vars)
        
        # Calculate summaries with privacy constraints
        summaries = []
        
        for name, group in grouped:
            if len(group) >= self.min_cell_size:
                summary = {'_'.join(group_vars): name}
                summary['n'] = len(group)
                
                for var in summary_vars:
                    if var in group.columns:
                        if group[var].dtype in ['float64', 'int64']:
                            summary[f'{var}_mean'] = group[var].mean()
                            summary[f'{var}_median'] = group[var].median()
                        else:
                            # For categorical, only report if dominant category
                            mode_count = group[var].value_counts().iloc[0]
                            if mode_count / len(group) > 0.8:
                                summary[f'{var}_mode'] = group[var].mode()[0]
                
                summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def create_audit_report(self) -> Dict[str, Any]:
        """Generate audit report of all queries"""
        
        report = {
            'total_queries': len(self.audit_log),
            'query_types': {},
            'timestamp_range': {
                'start': min(log['timestamp'] for log in self.audit_log) if self.audit_log else None,
                'end': max(log['timestamp'] for log in self.audit_log) if self.audit_log else None
            }
        }
        
        # Count query types
        for log in self.audit_log:
            action = log.get('action', 'unknown')
            report['query_types'][action] = report['query_types'].get(action, 0) + 1
        
        return report
    
    def validate_code_output(self, code: str) -> Tuple[bool, List[str]]:
        """Validate generated code for privacy compliance"""
        
        issues = []
        
        # Check for direct patient ID usage
        if re.search(r'patient_id|hospitalization_id|mrn', code, re.IGNORECASE):
            issues.append("Code contains potential patient identifiers")
        
        # Check for individual record access
        if re.search(r'\.iloc\[\d+\]|\.loc\[[\'\"]', code):
            issues.append("Code may access individual records")
        
        # Check for data export without aggregation
        if re.search(r'to_csv|to_excel|to_json', code) and not re.search(r'groupby|agg|mean|sum', code):
            issues.append("Code exports data without aggregation")
        
        # Check for external data transmission
        if re.search(r'requests\.|urllib|http|api', code):
            issues.append("Code contains external data transmission")
        
        return len(issues) == 0, issues


