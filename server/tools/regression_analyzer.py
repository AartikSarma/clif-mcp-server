"""
Regression Analysis Tool for CLIF MCP Server
Supports multiple regression model types with comprehensive output
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class RegressionAnalyzer:
    """Comprehensive regression analysis tool"""
    
    def fit_model(self, 
                  df: pd.DataFrame,
                  outcome: str,
                  predictors: List[str],
                  model_type: str,
                  interaction_terms: List[str] = None,
                  random_effects: List[str] = None) -> str:
        """Fit regression model and return formatted results"""
        
        # Check if outcome exists
        if outcome not in df.columns:
            return self._suggest_outcome_creation(outcome, df)
        
        # Check predictors
        missing_predictors = [p for p in predictors if p not in df.columns]
        if missing_predictors:
            return self._suggest_predictor_fixes(missing_predictors, df)
        
        # Prepare data
        df_clean = df[[outcome] + predictors].dropna()
        
        # Add interaction terms
        if interaction_terms:
            for term in interaction_terms:
                if '*' in term:
                    parts = term.split('*')
                    if all(p.strip() in df_clean.columns for p in parts):
                        df_clean[term] = df_clean[parts[0].strip()] * df_clean[parts[1].strip()]
                        predictors.append(term)
        
        # Fit appropriate model
        if model_type == "linear":
            return self._fit_linear(df_clean, outcome, predictors)
        elif model_type == "logistic":
            return self._fit_logistic(df_clean, outcome, predictors)
        elif model_type == "poisson":
            return self._fit_poisson(df_clean, outcome, predictors)
        elif model_type == "negative_binomial":
            return self._fit_negative_binomial(df_clean, outcome, predictors)
        elif model_type == "cox":
            return self._fit_cox(df_clean, outcome, predictors)
        elif model_type == "mixed_linear":
            return self._fit_mixed_linear(df_clean, outcome, predictors, random_effects)
        else:
            return f"Unknown model type: {model_type}"
    
    def _fit_linear(self, df: pd.DataFrame, outcome: str, predictors: List[str]) -> str:
        """Fit linear regression model"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            from scipy import stats
            
            X = df[predictors]
            y = df[outcome]
            
            # Add intercept
            X = pd.concat([pd.Series(1, index=X.index, name='intercept'), X], axis=1)
            
            # Fit model
            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)
            
            # Get predictions and residuals
            y_pred = model.predict(X)
            residuals = y - y_pred
            
            # Calculate statistics
            n = len(y)
            k = len(predictors)
            r2 = r2_score(y, y_pred)
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
            
            # Standard errors
            mse = np.mean(residuals ** 2)
            var_coef = mse * np.linalg.inv(X.T @ X).diagonal()
            se = np.sqrt(var_coef)
            
            # T-statistics and p-values
            t_stats = model.coef_ / se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
            
            # Format results
            result = f"""Linear Regression Results
========================================
Outcome: {outcome}
N observations: {n}
R-squared: {r2:.4f}
Adjusted R-squared: {adj_r2:.4f}

Coefficients:
"""
            for i, (coef_name, coef, stderr, t_stat, p_val) in enumerate(
                zip(['Intercept'] + predictors, model.coef_, se, t_stats, p_values)
            ):
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                result += f"  {coef_name:<20} {coef:>10.4f} ({stderr:>8.4f})  t={t_stat:>6.2f}  p={p_val:>6.4f} {sig}\n"
            
            result += "\nSignificance: *** p<0.001, ** p<0.01, * p<0.05"
            
            return result
            
        except ImportError:
            return "Linear regression requires scikit-learn. Please install: pip install scikit-learn scipy"
        except Exception as e:
            return f"Error fitting linear model: {str(e)}"
    
    def _fit_logistic(self, df: pd.DataFrame, outcome: str, predictors: List[str]) -> str:
        """Fit logistic regression model"""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            
            X = df[predictors]
            y = df[outcome]
            
            # Ensure binary outcome
            if y.nunique() > 2:
                return f"Logistic regression requires binary outcome. {outcome} has {y.nunique()} unique values."
            
            # Fit model
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            
            # Get predictions
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Calculate AUC
            auc = roc_auc_score(y, y_pred_proba)
            
            # Get coefficients and odds ratios
            coefs = model.coef_[0]
            odds_ratios = np.exp(coefs)
            
            # Format results
            result = f"""Logistic Regression Results
========================================
Outcome: {outcome}
N observations: {len(y)}
AUC: {auc:.4f}

Coefficients (Log Odds):
"""
            result += f"  {'Intercept':<20} {model.intercept_[0]:>10.4f}\n"
            
            for pred, coef, or_val in zip(predictors, coefs, odds_ratios):
                result += f"  {pred:<20} {coef:>10.4f}  (OR: {or_val:>6.2f})\n"
            
            result += "\nInterpretation: OR > 1 indicates increased odds, OR < 1 indicates decreased odds"
            
            return result
            
        except ImportError:
            return "Logistic regression requires scikit-learn. Please install: pip install scikit-learn"
        except Exception as e:
            return f"Error fitting logistic model: {str(e)}"
    
    def _fit_poisson(self, df: pd.DataFrame, outcome: str, predictors: List[str]) -> str:
        """Fit Poisson regression model"""
        try:
            import statsmodels.api as sm
            
            X = sm.add_constant(df[predictors])
            y = df[outcome]
            
            # Fit Poisson model
            model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
            
            # Format results
            result = f"""Poisson Regression Results
========================================
Outcome: {outcome}
N observations: {len(y)}
Log-Likelihood: {model.llf:.2f}
AIC: {model.aic:.2f}

Coefficients (Log Rate Ratios):
"""
            for param, coef, pval in zip(model.params.index, model.params.values, model.pvalues.values):
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                rr = np.exp(coef)
                result += f"  {param:<20} {coef:>10.4f}  (RR: {rr:>6.2f})  p={pval:>6.4f} {sig}\n"
            
            result += "\nSignificance: *** p<0.001, ** p<0.01, * p<0.05"
            result += "\nRR = Rate Ratio (exp(coefficient))"
            
            return result
            
        except ImportError:
            return "Poisson regression requires statsmodels. Please install: pip install statsmodels"
        except Exception as e:
            return f"Error fitting Poisson model: {str(e)}"
    
    def _fit_negative_binomial(self, df: pd.DataFrame, outcome: str, predictors: List[str]) -> str:
        """Fit negative binomial regression"""
        try:
            import statsmodels.api as sm
            
            X = sm.add_constant(df[predictors])
            y = df[outcome]
            
            # Fit negative binomial model
            model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
            
            result = f"""Negative Binomial Regression Results
========================================
Outcome: {outcome}
N observations: {len(y)}
Log-Likelihood: {model.llf:.2f}
AIC: {model.aic:.2f}

Coefficients (Log Rate Ratios):
"""
            for param, coef, pval in zip(model.params.index, model.params.values, model.pvalues.values):
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                rr = np.exp(coef)
                result += f"  {param:<20} {coef:>10.4f}  (RR: {rr:>6.2f})  p={pval:>6.4f} {sig}\n"
            
            result += "\nSignificance: *** p<0.001, ** p<0.01, * p<0.05"
            
            return result
            
        except ImportError:
            return "Negative binomial regression requires statsmodels. Please install: pip install statsmodels"
        except Exception as e:
            return f"Error fitting negative binomial model: {str(e)}"
    
    def _fit_cox(self, df: pd.DataFrame, outcome: str, predictors: List[str]) -> str:
        """Fit Cox proportional hazards model"""
        try:
            from lifelines import CoxPHFitter
            
            # For Cox model, we need time and event columns
            if outcome == "mortality":
                # Use LOS as time variable
                if "los_days" not in df.columns:
                    return "Cox model requires time variable. Please ensure 'los_days' is available."
                
                cox_df = df[predictors + ["los_days", outcome]].copy()
                cox_df = cox_df.rename(columns={"los_days": "T", outcome: "E"})
            else:
                return f"Cox model requires survival outcome. '{outcome}' may not be appropriate."
            
            # Fit model
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col='T', event_col='E')
            
            # Get results
            summary = cph.summary
            
            result = f"""Cox Proportional Hazards Results
========================================
Outcome: {outcome} (survival)
N observations: {len(df)}
N events: {cox_df['E'].sum()}
Concordance: {cph.concordance_index_:.4f}

Hazard Ratios:
"""
            for idx in summary.index:
                hr = summary.loc[idx, 'exp(coef)']
                p = summary.loc[idx, 'p']
                lower = summary.loc[idx, 'exp(coef) lower 95%']
                upper = summary.loc[idx, 'exp(coef) upper 95%']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                
                result += f"  {idx:<20} HR={hr:>6.2f} (95% CI: {lower:>5.2f}-{upper:>5.2f})  p={p:>6.4f} {sig}\n"
            
            result += "\nSignificance: *** p<0.001, ** p<0.01, * p<0.05"
            result += "\nHR > 1 indicates increased hazard, HR < 1 indicates decreased hazard"
            
            return result
            
        except ImportError:
            return "Cox regression requires lifelines. Please install: pip install lifelines"
        except Exception as e:
            return f"Error fitting Cox model: {str(e)}"
    
    def _fit_mixed_linear(self, df: pd.DataFrame, outcome: str, predictors: List[str], 
                         random_effects: List[str]) -> str:
        """Fit linear mixed effects model"""
        try:
            import statsmodels.formula.api as smf
            
            if not random_effects:
                return "Mixed models require random effects to be specified"
            
            # Build formula
            fixed_formula = f"{outcome} ~ " + " + ".join(predictors)
            
            # Note: statsmodels has limited mixed model support
            # For full functionality, would need R or specialized packages
            
            result = f"""Mixed Linear Model
========================================
Outcome: {outcome}
Fixed effects: {', '.join(predictors)}
Random effects: {', '.join(random_effects)}

Note: Full mixed model support requires additional packages.
Consider using R for comprehensive mixed model analysis.

For Python, you can install:
- pip install statsmodels (limited support)
- pip install pymer4 (R bridge)
"""
            
            return result
            
        except Exception as e:
            return f"Error with mixed model: {str(e)}"
    
    def _suggest_outcome_creation(self, outcome: str, df: pd.DataFrame) -> str:
        """Suggest how to create missing outcome variable"""
        
        suggestions = {
            'mortality': """
The 'mortality' variable is not found. You can create it from discharge_disposition:

df['mortality'] = (df['discharge_disposition'] == 'Expired').astype(int)
""",
            'los_days': """
The 'los_days' variable is not found. You can create it from admission/discharge times:

df['admission_dttm'] = pd.to_datetime(df['admission_dttm'])
df['discharge_dttm'] = pd.to_datetime(df['discharge_dttm'])
df['los_days'] = (df['discharge_dttm'] - df['admission_dttm']).dt.total_seconds() / 86400
""",
            'ventilator_days': """
The 'ventilator_days' variable requires respiratory_support data:

# Join with respiratory_support table and calculate duration
""",
            'icu_los': """
The 'icu_los' variable requires ADT (admission/discharge/transfer) data:

# Join with ADT table and sum ICU location durations
"""
        }
        
        if outcome in suggestions:
            return f"Outcome '{outcome}' not found in data.\n{suggestions[outcome]}"
        else:
            available = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
            return f"""Outcome '{outcome}' not found in data.

Available numeric columns that could be outcomes:
{', '.join(available[:20])}

You may need to:
1. Create a derived variable
2. Join with another table
3. Choose a different outcome variable
"""
    
    def _suggest_predictor_fixes(self, missing: List[str], df: pd.DataFrame) -> str:
        """Suggest fixes for missing predictors"""
        
        result = f"The following predictors are missing: {', '.join(missing)}\n\n"
        
        # Check for common issues
        for var in missing:
            if var in ['sex', 'race', 'ethnicity']:
                result += f"- '{var}' requires merging with patient table\n"
            elif var in ['age', 'age_at_admission']:
                result += f"- '{var}' can be calculated from birth_date and admission_dttm\n"
            elif 'vent' in var.lower():
                result += f"- '{var}' requires respiratory_support table\n"
            elif 'lab' in var.lower():
                result += f"- '{var}' requires labs table\n"
        
        # Show available columns
        available = list(df.columns)
        result += f"\nAvailable columns in current data:\n{', '.join(available[:30])}"
        
        if len(available) > 30:
            result += f"\n... and {len(available) - 30} more"
        
        return result