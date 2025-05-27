"""
Descriptive Statistics Calculator for CLIF MCP Server
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


class StatsCalculator:
    """Calculate descriptive statistics with stratification support"""
    
    def calculate_descriptive_stats(self, 
                                  df: pd.DataFrame, 
                                  variables: List[str],
                                  stratify_by: Optional[str] = None) -> str:
        """Calculate comprehensive descriptive statistics"""
        
        # Check variables exist
        missing_vars = [v for v in variables if v not in df.columns]
        if missing_vars:
            available = list(df.columns)
            return f"""Variables not found: {', '.join(missing_vars)}

Available columns:
{', '.join(available[:50])}
"""
        
        result = "Descriptive Statistics\n"
        result += "=" * 50 + "\n\n"
        
        if stratify_by and stratify_by in df.columns:
            # Stratified analysis
            result += f"Stratified by: {stratify_by}\n"
            result += "-" * 30 + "\n\n"
            
            groups = df.groupby(stratify_by)
            
            for group_name, group_df in groups:
                result += f"\n{stratify_by} = {group_name} (n={len(group_df)})\n"
                result += self._calculate_stats_for_group(group_df, variables)
                result += "\n"
        else:
            # Overall analysis
            result += f"Overall (n={len(df)})\n"
            result += self._calculate_stats_for_group(df, variables)
        
        return result
    
    def _calculate_stats_for_group(self, df: pd.DataFrame, variables: List[str]) -> str:
        """Calculate statistics for a single group"""
        
        result = ""
        
        for var in variables:
            if var not in df.columns:
                continue
                
            result += f"\n{var}:\n"
            
            # Get variable type
            dtype = df[var].dtype
            
            if dtype in ['float64', 'int64']:
                # Numeric variable
                stats = self._numeric_stats(df[var])
                result += f"  Mean (SD): {stats['mean']:.2f} ({stats['std']:.2f})\n"
                result += f"  Median [IQR]: {stats['median']:.2f} [{stats['q1']:.2f}-{stats['q3']:.2f}]\n"
                result += f"  Range: {stats['min']:.2f} - {stats['max']:.2f}\n"
                result += f"  Missing: {stats['missing']} ({stats['missing_pct']:.1f}%)\n"
                
            elif dtype == 'object' or df[var].nunique() < 20:
                # Categorical variable
                stats = self._categorical_stats(df[var])
                result += "  Frequency (%):\n"
                
                # Show top categories
                for cat, count, pct in stats['frequencies'][:10]:
                    result += f"    {cat}: {count} ({pct:.1f}%)\n"
                
                if len(stats['frequencies']) > 10:
                    result += f"    ... and {len(stats['frequencies']) - 10} more categories\n"
                
                result += f"  Missing: {stats['missing']} ({stats['missing_pct']:.1f}%)\n"
                
            elif dtype == 'datetime64[ns]':
                # DateTime variable
                stats = self._datetime_stats(df[var])
                result += f"  Earliest: {stats['min']}\n"
                result += f"  Latest: {stats['max']}\n"
                result += f"  Range: {stats['range_days']:.1f} days\n"
                result += f"  Missing: {stats['missing']} ({stats['missing_pct']:.1f}%)\n"
        
        return result
    
    def _numeric_stats(self, series: pd.Series) -> Dict[str, float]:
        """Calculate statistics for numeric variable"""
        
        valid = series.dropna()
        
        return {
            'mean': valid.mean(),
            'std': valid.std(),
            'median': valid.median(),
            'q1': valid.quantile(0.25),
            'q3': valid.quantile(0.75),
            'min': valid.min(),
            'max': valid.max(),
            'missing': series.isna().sum(),
            'missing_pct': 100 * series.isna().sum() / len(series)
        }
    
    def _categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for categorical variable"""
        
        # Get value counts
        counts = series.value_counts()
        total_valid = counts.sum()
        
        # Calculate frequencies
        frequencies = []
        for value, count in counts.items():
            pct = 100 * count / total_valid
            frequencies.append((value, count, pct))
        
        return {
            'frequencies': frequencies,
            'n_categories': len(counts),
            'missing': series.isna().sum(),
            'missing_pct': 100 * series.isna().sum() / len(series)
        }
    
    def _datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for datetime variable"""
        
        valid = series.dropna()
        
        if len(valid) == 0:
            return {
                'min': 'N/A',
                'max': 'N/A',
                'range_days': 0,
                'missing': len(series),
                'missing_pct': 100.0
            }
        
        min_date = valid.min()
        max_date = valid.max()
        range_days = (max_date - min_date).total_seconds() / 86400
        
        return {
            'min': min_date.strftime('%Y-%m-%d %H:%M'),
            'max': max_date.strftime('%Y-%m-%d %H:%M'),
            'range_days': range_days,
            'missing': series.isna().sum(),
            'missing_pct': 100 * series.isna().sum() / len(series)
        }
    
    def compare_groups(self, 
                      df: pd.DataFrame,
                      variable: str,
                      group_var: str) -> str:
        """Compare a variable across groups with statistical tests"""
        
        if variable not in df.columns or group_var not in df.columns:
            return "Variable or grouping variable not found"
        
        result = f"\nComparison of '{variable}' by '{group_var}'\n"
        result += "=" * 50 + "\n"
        
        # Determine variable type
        if df[variable].dtype in ['float64', 'int64']:
            # Numeric comparison
            result += self._compare_numeric(df, variable, group_var)
        else:
            # Categorical comparison
            result += self._compare_categorical(df, variable, group_var)
        
        return result
    
    def _compare_numeric(self, df: pd.DataFrame, variable: str, group_var: str) -> str:
        """Compare numeric variable across groups"""
        
        try:
            from scipy import stats as scipy_stats
            
            groups = df.groupby(group_var)[variable]
            
            result = "\nGroup Statistics:\n"
            
            group_data = []
            for name, group in groups:
                valid = group.dropna()
                result += f"\n{group_var} = {name}:\n"
                result += f"  n = {len(valid)}\n"
                result += f"  Mean (SD) = {valid.mean():.2f} ({valid.std():.2f})\n"
                result += f"  Median [IQR] = {valid.median():.2f} [{valid.quantile(0.25):.2f}-{valid.quantile(0.75):.2f}]\n"
                
                group_data.append(valid)
            
            # Statistical test
            if len(group_data) == 2:
                # T-test for 2 groups
                t_stat, p_val = scipy_stats.ttest_ind(group_data[0], group_data[1])
                result += f"\nTwo-sample t-test: t = {t_stat:.3f}, p = {p_val:.4f}\n"
                
                # Mann-Whitney U test
                u_stat, p_val_mw = scipy_stats.mannwhitneyu(group_data[0], group_data[1])
                result += f"Mann-Whitney U test: U = {u_stat:.1f}, p = {p_val_mw:.4f}\n"
                
            elif len(group_data) > 2:
                # ANOVA for multiple groups
                f_stat, p_val = scipy_stats.f_oneway(*group_data)
                result += f"\nOne-way ANOVA: F = {f_stat:.3f}, p = {p_val:.4f}\n"
                
                # Kruskal-Wallis test
                h_stat, p_val_kw = scipy_stats.kruskal(*group_data)
                result += f"Kruskal-Wallis test: H = {h_stat:.3f}, p = {p_val_kw:.4f}\n"
            
            return result
            
        except ImportError:
            return "\nStatistical tests require scipy. Please install: pip install scipy"
        except Exception as e:
            return f"\nError in comparison: {str(e)}"
    
    def _compare_categorical(self, df: pd.DataFrame, variable: str, group_var: str) -> str:
        """Compare categorical variable across groups"""
        
        try:
            from scipy.stats import chi2_contingency
            
            # Create contingency table
            ct = pd.crosstab(df[variable], df[group_var])
            
            result = "\nContingency Table:\n"
            result += str(ct) + "\n"
            
            # Proportions
            result += "\nRow Proportions (%):\n"
            prop_table = pd.crosstab(df[variable], df[group_var], normalize='columns') * 100
            result += prop_table.round(1).to_string() + "\n"
            
            # Chi-square test
            chi2, p_val, dof, expected = chi2_contingency(ct)
            result += f"\nChi-square test: χ² = {chi2:.3f}, df = {dof}, p = {p_val:.4f}\n"
            
            return result
            
        except ImportError:
            return "\nStatistical tests require scipy. Please install: pip install scipy"
        except Exception as e:
            return f"\nError in comparison: {str(e)}"