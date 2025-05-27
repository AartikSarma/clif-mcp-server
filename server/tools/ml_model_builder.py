"""
Machine Learning Model Builder for CLIF MCP Server
Supports various ML models for clinical prediction tasks
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    brier_score_loss
)
try:
    from sklearn.calibration import calibration_curve, CalibratedClassifierCV
except ImportError:
    # Older sklearn versions might not have calibration_curve
    calibration_curve = None
    from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR

# Advanced ML
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Survival analysis
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

# Imbalanced learning
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


class MLModelBuilder:
    def __init__(self, data_path: str = None):
        self.data_path = Path(data_path) if data_path else None
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.shap_values = {}
        
    def prepare_features(self, 
                        df: pd.DataFrame,
                        feature_list: List[str],
                        outcome: str,
                        task: str,
                        handle_missing: str = 'impute',
                        scale_features: bool = True,
                        create_interactions: bool = False,
                        temporal_features: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare features for ML modeling"""
        
        # Extract features and outcome
        df_work = df.copy()
        
        # Create temporal features if specified
        if temporal_features:
            df_work = self._create_temporal_features(df_work, temporal_features)
            # Add new temporal features to feature list
            feature_list.extend([f for f in df_work.columns if f.startswith('temporal_') and f not in feature_list])
        
        # Handle missing data
        if handle_missing == 'drop':
            df_work = df_work.dropna(subset=feature_list + [outcome])
        elif handle_missing == 'impute':
            # Separate numeric and categorical
            numeric_features = df_work[feature_list].select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = df_work[feature_list].select_dtypes(include=['object']).columns.tolist()
            
            if numeric_features:
                imputer = SimpleImputer(strategy='median')
                df_work[numeric_features] = imputer.fit_transform(df_work[numeric_features])
            
            if categorical_features:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                df_work[categorical_features] = imputer_cat.fit_transform(df_work[categorical_features])
        elif handle_missing == 'indicator':
            # Add missingness indicators
            for col in feature_list:
                if df_work[col].isnull().any():
                    df_work[f'{col}_missing'] = df_work[col].isnull().astype(int)
                    feature_list.append(f'{col}_missing')
            
            # Then impute
            df_work = df_work.fillna(df_work.median())
        
        # Create dummy variables for categorical features
        categorical_features = df_work[feature_list].select_dtypes(include=['object']).columns.tolist()
        if categorical_features:
            df_work = pd.get_dummies(df_work, columns=categorical_features, drop_first=True)
            # Update feature list
            feature_list = [f for f in df_work.columns if f != outcome and any(f.startswith(cat) for cat in feature_list)]
        
        # Create interaction terms if requested
        if create_interactions:
            numeric_features = [f for f in feature_list if f in df_work.columns and df_work[f].dtype in [np.float64, np.int64]]
            for i, feat1 in enumerate(numeric_features):
                for feat2 in numeric_features[i+1:]:
                    interaction_name = f"{feat1}_x_{feat2}"
                    df_work[interaction_name] = df_work[feat1] * df_work[feat2]
                    feature_list.append(interaction_name)
        
        # Extract final features and outcome
        X = df_work[feature_list]
        y = df_work[outcome]
        
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            X = X_scaled
        
        return X, y, feature_list
    
    def _create_temporal_features(self, df: pd.DataFrame, temporal_config: Dict[str, Any]) -> pd.DataFrame:
        """Create temporal features from time-series data"""
        
        # Example temporal features:
        # - Trends (slope over time)
        # - Variability (std dev)
        # - Change from baseline
        # - Time since event
        
        for feature_name, config in temporal_config.items():
            source_col = config.get('source_column')
            feature_type = config.get('type', 'trend')
            window = config.get('window', 24)  # hours
            
            if feature_type == 'trend':
                # Calculate linear trend over window
                df[f'temporal_trend_{source_col}_{window}h'] = np.nan  # Placeholder
            elif feature_type == 'variability':
                # Calculate std dev over window
                df[f'temporal_std_{source_col}_{window}h'] = np.nan  # Placeholder
            elif feature_type == 'change_from_baseline':
                # Calculate change from first value
                df[f'temporal_change_{source_col}'] = np.nan  # Placeholder
        
        return df
    
    def fit_model(self,
                  df: pd.DataFrame,
                  model_type: str,
                  task: str,
                  outcome: str,
                  features: List[str],
                  validation_strategy: str = 'cv',
                  test_size: float = 0.2,
                  cv_folds: int = 5,
                  hyperparameter_tuning: bool = False,
                  handle_imbalance: str = 'none',
                  handle_missing: str = 'impute',
                  scale_features: bool = True,
                  calibrate: bool = False,
                  interpretability: bool = True,
                  random_state: int = 42) -> Dict[str, Any]:
        """Fit ML model with specified configuration"""
        
        # Prepare features
        X, y, feature_names = self.prepare_features(
            df, features, outcome, task, handle_missing, scale_features
        )
        
        # Handle class imbalance for classification
        if task == 'classification' and handle_imbalance != 'none':
            X, y = self._handle_imbalance(X, y, method=handle_imbalance, random_state=random_state)
        
        # Split data based on validation strategy
        if validation_strategy == 'holdout':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y if task == 'classification' else None
            )
        elif validation_strategy == 'temporal':
            # Temporal split - use last portion as test
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:  # cv
            X_train, X_test = X, X
            y_train, y_test = y, y
        
        # Get model
        model = self._get_model(model_type, task, random_state)
        
        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            model = self._tune_hyperparameters(
                model, X_train, y_train, task, cv_folds, random_state
            )
        
        # Fit model
        if validation_strategy == 'cv':
            # Cross-validation
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state) if task == 'classification' else cv_folds
            scores = cross_val_score(
                model, X_train, y_train, cv=cv,
                scoring='roc_auc' if task == 'classification' else 'neg_mean_squared_error'
            )
            model.fit(X_train, y_train)  # Fit on full training data
        else:
            model.fit(X_train, y_train)
        
        # Calibrate if requested (classification only)
        if task == 'classification' and calibrate:
            model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
            model.fit(X_train, y_train)
        
        # Evaluate model
        results = self._evaluate_model(
            model, X_test, y_test, X_train, y_train, task, feature_names
        )
        
        # Add interpretability
        if interpretability:
            self._add_interpretability(model, X_train, feature_names, task)
        
        # Store model and results
        model_id = f"{model_type}_{outcome}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.models[model_id] = {
            'model': model,
            'feature_names': feature_names,
            'task': task,
            'outcome': outcome,
            'model_type': model_type
        }
        self.results[model_id] = results
        
        return {
            'model_id': model_id,
            'results': results,
            'feature_importance': self.feature_importance.get(model_id, {}),
            'model_info': {
                'type': model_type,
                'task': task,
                'outcome': outcome,
                'n_features': len(feature_names),
                'n_samples_train': len(X_train),
                'n_samples_test': len(X_test)
            }
        }
    
    def _get_model(self, model_type: str, task: str, random_state: int):
        """Get model instance based on type and task"""
        
        if task == 'classification':
            if model_type == 'logistic_regression':
                return LogisticRegression(random_state=random_state, max_iter=1000)
            elif model_type == 'random_forest':
                return RandomForestClassifier(n_estimators=100, random_state=random_state)
            elif model_type == 'xgboost' and HAS_XGBOOST:
                return xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
            elif model_type == 'lightgbm' and HAS_LIGHTGBM:
                return lgb.LGBMClassifier(random_state=random_state, verbosity=-1)
            elif model_type == 'svm':
                return SVC(probability=True, random_state=random_state)
            else:
                raise ValueError(f"Unknown classification model type: {model_type}")
                
        elif task == 'regression':
            if model_type == 'linear_regression':
                return Ridge(random_state=random_state)
            elif model_type == 'random_forest':
                return RandomForestRegressor(n_estimators=100, random_state=random_state)
            elif model_type == 'xgboost' and HAS_XGBOOST:
                return xgb.XGBRegressor(random_state=random_state)
            elif model_type == 'lightgbm' and HAS_LIGHTGBM:
                return lgb.LGBMRegressor(random_state=random_state, verbosity=-1)
            elif model_type == 'svm':
                return SVR()
            else:
                raise ValueError(f"Unknown regression model type: {model_type}")
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _handle_imbalance(self, X: pd.DataFrame, y: pd.Series, method: str, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance"""
        
        if not HAS_IMBLEARN:
            print("Warning: imblearn not installed, skipping imbalance handling")
            return X, y
        
        if method == 'smote':
            sampler = SMOTE(random_state=random_state)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=random_state)
        else:
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        return X_resampled, y_resampled
    
    def _tune_hyperparameters(self, model, X_train, y_train, task, cv_folds, random_state):
        """Tune hyperparameters using grid search"""
        
        # Define parameter grids
        param_grids = {
            'RandomForestClassifier': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'RandomForestRegressor': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'XGBClassifier': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            },
            'XGBRegressor': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            },
            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        }
        
        model_class = type(model).__name__
        if model_class in param_grids:
            grid_search = GridSearchCV(
                model,
                param_grids[model_class],
                cv=cv_folds,
                scoring='roc_auc' if task == 'classification' else 'neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        
        return model
    
    def _evaluate_model(self, model, X_test, y_test, X_train, y_train, task, feature_names):
        """Evaluate model performance"""
        
        results = {
            'train_scores': {},
            'test_scores': {},
            'predictions': {},
            'feature_names': feature_names
        }
        
        if task == 'classification':
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_pred_proba_train = model.predict_proba(X_train)[:, 1]
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            results['train_scores'] = {
                'accuracy': accuracy_score(y_train, y_pred_train),
                'auc_roc': roc_auc_score(y_train, y_pred_proba_train),
                'auc_pr': average_precision_score(y_train, y_pred_proba_train),
                'brier_score': brier_score_loss(y_train, y_pred_proba_train)
            }
            
            results['test_scores'] = {
                'accuracy': accuracy_score(y_test, y_pred_test),
                'auc_roc': roc_auc_score(y_test, y_pred_proba_test),
                'auc_pr': average_precision_score(y_test, y_pred_proba_test),
                'brier_score': brier_score_loss(y_test, y_pred_proba_test)
            }
            
            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred_test).tolist()
            
            # ROC and PR curves
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_test)
            
            results['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            results['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
            
            # Calibration (if available)
            if calibration_curve is not None:
                fraction_pos, mean_pred = calibration_curve(y_test, y_pred_proba_test, n_bins=10)
                results['calibration_curve'] = {
                    'fraction_positive': fraction_pos.tolist(),
                    'mean_predicted': mean_pred.tolist()
                }
            
        elif task == 'regression':
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            results['train_scores'] = {
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train)
            }
            
            results['test_scores'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test)
            }
            
            # Residuals
            residuals = y_test - y_pred_test
            results['residuals'] = {
                'values': residuals.tolist(),
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals))
            }
        
        return results
    
    def _add_interpretability(self, model, X_train, feature_names, task):
        """Add model interpretability"""
        
        model_type = type(model).__name__
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = dict(zip(feature_names, importance))
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            self.feature_importance[model_type] = feature_importance
        
        # SHAP values if available
        if HAS_SHAP and len(X_train) < 10000:  # Limit for computational reasons
            try:
                if model_type in ['RandomForestClassifier', 'RandomForestRegressor', 'XGBClassifier', 'XGBRegressor']:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_train)
                    
                    # For classification, take values for positive class
                    if task == 'classification' and len(shap_values) == 2:
                        shap_values = shap_values[1]
                    
                    # Store summary
                    self.shap_values[model_type] = {
                        'values': shap_values[:100].tolist() if hasattr(shap_values, 'tolist') else shap_values[:100],  # Limit size
                        'base_value': float(explainer.expected_value[1] if task == 'classification' else explainer.expected_value),
                        'feature_names': feature_names
                    }
            except Exception as e:
                print(f"Could not compute SHAP values: {e}")
    
    def predict(self, model_id: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with a trained model"""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        model = model_info['model']
        
        # Ensure features match
        expected_features = model_info['feature_names']
        X = X[expected_features]
        
        return model.predict(X)
    
    def predict_proba(self, model_id: str, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities for classification models"""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        if model_info['task'] != 'classification':
            raise ValueError("predict_proba only available for classification models")
        
        model = model_info['model']
        expected_features = model_info['feature_names']
        X = X[expected_features]
        
        return model.predict_proba(X)
    
    def save_model(self, model_id: str, filepath: str):
        """Save model to disk"""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_data = {
            'model': self.models[model_id],
            'results': self.results.get(model_id, {}),
            'feature_importance': self.feature_importance.get(model_id, {}),
            'shap_values': self.shap_values.get(model_id, {})
        }
        
        joblib.dump(model_data, filepath)
        
    def load_model(self, filepath: str) -> str:
        """Load model from disk"""
        
        model_data = joblib.load(filepath)
        
        # Generate new model_id
        model_id = f"loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.models[model_id] = model_data['model']
        self.results[model_id] = model_data.get('results', {})
        self.feature_importance[model_id] = model_data.get('feature_importance', {})
        self.shap_values[model_id] = model_data.get('shap_values', {})
        
        return model_id
    
    def get_clinical_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard clinical risk scores"""
        
        scores = pd.DataFrame(index=df.index)
        
        # SOFA Score (simplified)
        if all(col in df.columns for col in ['pao2_fio2_ratio', 'platelets', 'bilirubin', 'map', 'gcs', 'creatinine']):
            scores['sofa_score'] = self._calculate_sofa(df)
        
        # APACHE II (simplified)
        if all(col in df.columns for col in ['temperature', 'map', 'heart_rate', 'respiratory_rate', 'pao2', 'ph', 'sodium', 'potassium', 'creatinine', 'hematocrit', 'wbc', 'gcs', 'age']):
            scores['apache_ii'] = self._calculate_apache_ii(df)
        
        return scores
    
    def _calculate_sofa(self, df: pd.DataFrame) -> pd.Series:
        """Calculate simplified SOFA score"""
        sofa = pd.Series(0, index=df.index)
        
        # Respiration
        pf_ratio = df.get('pao2_fio2_ratio', 400)
        sofa += (pf_ratio < 400).astype(int)
        sofa += (pf_ratio < 300).astype(int)
        sofa += (pf_ratio < 200).astype(int)
        sofa += (pf_ratio < 100).astype(int)
        
        # Coagulation
        platelets = df.get('platelets', 150)
        sofa += (platelets < 150).astype(int)
        sofa += (platelets < 100).astype(int)
        sofa += (platelets < 50).astype(int)
        sofa += (platelets < 20).astype(int)
        
        # Add other components similarly...
        
        return sofa
    
    def _calculate_apache_ii(self, df: pd.DataFrame) -> pd.Series:
        """Calculate simplified APACHE II score"""
        apache = pd.Series(0, index=df.index)
        
        # Temperature points
        temp = df.get('temperature', 37)
        apache += np.where(temp >= 41, 4, 0)
        apache += np.where((temp >= 39) & (temp < 41), 3, 0)
        
        # Add other components similarly...
        
        return apache