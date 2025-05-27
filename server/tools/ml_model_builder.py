"""
Machine Learning Model Builder for CLIF MCP Server
Flexible ML tool for clinical prediction with user-defined outcomes and features
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

# Core ML imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    KFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    # Classification metrics
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report, brier_score_loss,
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

# Models
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Feature selection
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    RFE, RFECV, SelectFromModel,
    chi2, mutual_info_classif, mutual_info_regression
)

# Optional advanced libraries
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
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False


class MLModelBuilder:
    """Flexible ML model builder for clinical outcomes prediction"""
    
    # Available models by task
    CLASSIFICATION_MODELS = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'extra_trees': ExtraTreesClassifier,
        'svm': SVC,
        'knn': KNeighborsClassifier,
        'naive_bayes': GaussianNB,
        'decision_tree': DecisionTreeClassifier,
        'adaboost': AdaBoostClassifier,
        'sgd': SGDClassifier
    }
    
    REGRESSION_MODELS = {
        'linear_regression': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'extra_trees': ExtraTreesRegressor,
        'svm': SVR,
        'knn': KNeighborsRegressor,
        'decision_tree': DecisionTreeRegressor,
        'adaboost': AdaBoostRegressor,
        'sgd': SGDRegressor
    }
    
    def __init__(self, data_path: str = None):
        self.data_path = Path(data_path) if data_path else None
        self.models = {}
        self.results = {}
        self.preprocessors = {}
        
        # Add optional models if available
        if HAS_XGBOOST:
            self.CLASSIFICATION_MODELS['xgboost'] = xgb.XGBClassifier
            self.REGRESSION_MODELS['xgboost'] = xgb.XGBRegressor
        
        if HAS_LIGHTGBM:
            self.CLASSIFICATION_MODELS['lightgbm'] = lgb.LGBMClassifier
            self.REGRESSION_MODELS['lightgbm'] = lgb.LGBMRegressor
            
        if HAS_NEURAL:
            self.CLASSIFICATION_MODELS['neural_network'] = MLPClassifier
            self.REGRESSION_MODELS['neural_network'] = MLPRegressor
    
    def build_model(self,
                   df: pd.DataFrame,
                   outcome: str,
                   features: List[str],
                   model_type: str = 'auto',
                   task: str = 'auto',
                   validation_split: float = 0.2,
                   cv_folds: int = 5,
                   feature_selection: Optional[Dict[str, Any]] = None,
                   preprocessing: Optional[Dict[str, Any]] = None,
                   hyperparameter_tuning: bool = True,
                   random_state: int = 42) -> Dict[str, Any]:
        """
        Build ML model with flexible configuration
        
        Args:
            df: Input dataframe
            outcome: Target variable name
            features: List of feature column names
            model_type: Model to use ('auto' for automatic selection)
            task: 'classification' or 'regression' ('auto' to infer)
            validation_split: Test set size
            cv_folds: Number of CV folds
            feature_selection: Feature selection config
            preprocessing: Preprocessing config
            hyperparameter_tuning: Whether to tune hyperparameters
            random_state: Random seed
        
        Returns:
            Dictionary with model results and metadata
        """
        
        # Infer task if auto
        if task == 'auto':
            task = self._infer_task(df[outcome])
        
        # Prepare data
        X, y, feature_names, preprocessor = self._prepare_data(
            df, features, outcome, preprocessing, task
        )
        
        # Feature selection if requested
        if feature_selection:
            X, feature_names = self._select_features(
                X, y, feature_names, task, feature_selection
            )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=random_state,
            stratify=y if task == 'classification' else None
        )
        
        # Select model
        if model_type == 'auto':
            model_type = self._select_best_model(X_train, y_train, task, cv_folds)
        
        # Get and configure model
        model = self._get_model(model_type, task, random_state)
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            model = self._tune_hyperparameters(
                model, X_train, y_train, task, cv_folds
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        results = self._evaluate_model(
            model, X_train, X_test, y_train, y_test, task
        )
        
        # Add feature importance
        feature_importance = self._get_feature_importance(
            model, feature_names, X_train, y_train
        )
        
        # Store model
        model_id = f"{model_type}_{outcome}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.models[model_id] = {
            'model': model,
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'task': task,
            'outcome': outcome,
            'model_type': model_type
        }
        self.results[model_id] = results
        
        return {
            'model_id': model_id,
            'model_type': model_type,
            'task': task,
            'performance': {
                'train': results['train_scores'],
                'test': results['test_scores']
            },
            'feature_importance': feature_importance,
            'n_samples': len(X),
            'n_features': len(feature_names),
            'outcome_distribution': {
                'unique_values': y.nunique() if task == 'classification' else None,
                'mean': float(y.mean()) if task == 'regression' else None,
                'std': float(y.std()) if task == 'regression' else None
            }
        }
    
    def _infer_task(self, y: pd.Series) -> str:
        """Infer whether task is classification or regression"""
        # Check if binary
        if y.nunique() == 2:
            return 'classification'
        
        # Check if integer with few unique values (likely classification)
        if y.dtype in ['int64', 'int32'] and y.nunique() < 10:
            return 'classification'
        
        # Check if categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            return 'classification'
        
        # Otherwise assume regression
        return 'regression'
    
    def _prepare_data(self, 
                     df: pd.DataFrame,
                     features: List[str],
                     outcome: str,
                     preprocessing: Optional[Dict[str, Any]],
                     task: str) -> Tuple[np.ndarray, pd.Series, List[str], Any]:
        """Prepare data for modeling"""
        
        # Default preprocessing
        if preprocessing is None:
            preprocessing = {
                'handle_missing': 'impute',
                'scaling': 'standard',
                'encode_categorical': True
            }
        
        df_work = df[features + [outcome]].copy()
        
        # Handle missing values
        if preprocessing.get('handle_missing') == 'drop':
            df_work = df_work.dropna()
        elif preprocessing.get('handle_missing') == 'impute':
            # Separate numeric and categorical
            numeric_cols = df_work[features].select_dtypes(include=[np.number]).columns
            categorical_cols = df_work[features].select_dtypes(include=['object', 'category']).columns
            
            # Impute numeric
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='median')
                df_work[numeric_cols] = imputer.fit_transform(df_work[numeric_cols])
            
            # Impute categorical
            if len(categorical_cols) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                df_work[categorical_cols] = imputer_cat.fit_transform(df_work[categorical_cols])
        
        # Encode categorical variables
        if preprocessing.get('encode_categorical', True):
            categorical_cols = df_work[features].select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                df_work = pd.get_dummies(df_work, columns=categorical_cols, drop_first=True)
                # Update feature list
                features = [col for col in df_work.columns if col != outcome]
        
        # Extract features and target
        X = df_work[features].values
        y = df_work[outcome]
        
        # Scale features
        scaler = None
        scaling = preprocessing.get('scaling', 'standard')
        if scaling == 'standard':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaling == 'minmax':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        elif scaling == 'robust':
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
        
        return X, y, features, scaler
    
    def _select_features(self,
                        X: np.ndarray,
                        y: pd.Series,
                        feature_names: List[str],
                        task: str,
                        config: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Select features based on configuration"""
        
        method = config.get('method', 'kbest')
        n_features = config.get('n_features', 10)
        
        if method == 'kbest':
            if task == 'classification':
                selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
            else:
                selector = SelectKBest(f_regression, k=min(n_features, X.shape[1]))
        
        elif method == 'mutual_info':
            if task == 'classification':
                selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
            else:
                selector = SelectKBest(mutual_info_regression, k=min(n_features, X.shape[1]))
        
        elif method == 'rfe':
            base_estimator = RandomForestClassifier(n_estimators=50) if task == 'classification' else RandomForestRegressor(n_estimators=50)
            selector = RFE(base_estimator, n_features_to_select=n_features)
        
        else:
            return X, feature_names
        
        # Fit selector
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        mask = selector.get_support()
        selected_features = [f for f, m in zip(feature_names, mask) if m]
        
        return X_selected, selected_features
    
    def _select_best_model(self, X: np.ndarray, y: pd.Series, task: str, cv_folds: int) -> str:
        """Automatically select best model based on CV performance"""
        
        models_to_try = {}
        if task == 'classification':
            models_to_try = {
                'logistic_regression': LogisticRegression(max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100)
            }
            scoring = 'roc_auc' if y.nunique() == 2 else 'f1_weighted'
        else:
            models_to_try = {
                'linear_regression': Ridge(),
                'random_forest': RandomForestRegressor(n_estimators=100),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100)
            }
            scoring = 'neg_mean_squared_error'
        
        best_score = -np.inf
        best_model = 'random_forest'
        
        for name, model in models_to_try.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
                avg_score = scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = name
            except:
                continue
        
        return best_model
    
    def _get_model(self, model_type: str, task: str, random_state: int):
        """Get model instance"""
        
        if task == 'classification':
            model_class = self.CLASSIFICATION_MODELS.get(model_type)
        else:
            model_class = self.REGRESSION_MODELS.get(model_type)
        
        if model_class is None:
            raise ValueError(f"Unknown model type: {model_type} for task: {task}")
        
        # Configure model with sensible defaults
        kwargs = {'random_state': random_state} if 'random_state' in model_class.__init__.__code__.co_varnames else {}
        
        # Special configurations
        if model_type == 'svm':
            kwargs['probability'] = True if task == 'classification' else False
        elif model_type in ['xgboost', 'lightgbm']:
            kwargs['verbosity'] = 0
        elif model_type == 'neural_network':
            kwargs['hidden_layer_sizes'] = (100, 50)
            kwargs['max_iter'] = 1000
        
        return model_class(**kwargs)
    
    def _tune_hyperparameters(self, model, X: np.ndarray, y: pd.Series, task: str, cv_folds: int):
        """Tune hyperparameters using GridSearchCV"""
        
        # Define parameter grids for common models
        param_grids = {
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'RandomForestRegressor': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'GradientBoostingClassifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'GradientBoostingRegressor': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l2']
            },
            'Ridge': {
                'alpha': [0.001, 0.01, 0.1, 1, 10]
            }
        }
        
        model_name = type(model).__name__
        if model_name not in param_grids:
            return model
        
        scoring = 'roc_auc' if task == 'classification' and y.nunique() == 2 else None
        
        grid_search = GridSearchCV(
            model,
            param_grids[model_name],
            cv=min(cv_folds, 3),  # Use fewer folds for speed
            scoring=scoring,
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    
    def _evaluate_model(self, model, X_train, X_test, y_train, y_test, task):
        """Comprehensive model evaluation"""
        
        results = {
            'train_scores': {},
            'test_scores': {}
        }
        
        if task == 'classification':
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Probabilities (if available)
            if hasattr(model, 'predict_proba'):
                y_proba_train = model.predict_proba(X_train)
                y_proba_test = model.predict_proba(X_test)
                
                # For binary classification
                if y_train.nunique() == 2:
                    results['train_scores']['auc_roc'] = roc_auc_score(y_train, y_proba_train[:, 1])
                    results['test_scores']['auc_roc'] = roc_auc_score(y_test, y_proba_test[:, 1])
                    results['train_scores']['auc_pr'] = average_precision_score(y_train, y_proba_train[:, 1])
                    results['test_scores']['auc_pr'] = average_precision_score(y_test, y_proba_test[:, 1])
            
            # Standard metrics
            results['train_scores']['accuracy'] = accuracy_score(y_train, y_pred_train)
            results['test_scores']['accuracy'] = accuracy_score(y_test, y_pred_test)
            results['train_scores']['precision'] = precision_score(y_train, y_pred_train, average='weighted')
            results['test_scores']['precision'] = precision_score(y_test, y_pred_test, average='weighted')
            results['train_scores']['recall'] = recall_score(y_train, y_pred_train, average='weighted')
            results['test_scores']['recall'] = recall_score(y_test, y_pred_test, average='weighted')
            results['train_scores']['f1'] = f1_score(y_train, y_pred_train, average='weighted')
            results['test_scores']['f1'] = f1_score(y_test, y_pred_test, average='weighted')
            
            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred_test).tolist()
            
        else:  # Regression
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            results['train_scores']['rmse'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
            results['test_scores']['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred_test))
            results['train_scores']['mae'] = mean_absolute_error(y_train, y_pred_train)
            results['test_scores']['mae'] = mean_absolute_error(y_test, y_pred_test)
            results['train_scores']['r2'] = r2_score(y_train, y_pred_train)
            results['test_scores']['r2'] = r2_score(y_test, y_pred_test)
            results['train_scores']['explained_variance'] = explained_variance_score(y_train, y_pred_train)
            results['test_scores']['explained_variance'] = explained_variance_score(y_test, y_pred_test)
        
        return results
    
    def _get_feature_importance(self, model, feature_names: List[str], X: np.ndarray, y: pd.Series) -> Dict[str, float]:
        """Extract feature importance from model"""
        
        importance_dict = {}
        
        # Tree-based models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_dict = dict(zip(feature_names, importances))
        
        # Linear models
        elif hasattr(model, 'coef_'):
            # For binary classification, coef_ might be 2D
            coef = model.coef_
            if coef.ndim > 1:
                coef = coef[0]
            importance_dict = dict(zip(feature_names, np.abs(coef)))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def predict(self, model_id: str, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained model"""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        model = model_info['model']
        preprocessor = model_info.get('preprocessor')
        feature_names = model_info['feature_names']
        
        # Prepare features
        X = df[feature_names].values
        
        # Apply preprocessing if available
        if preprocessor:
            X = preprocessor.transform(X)
        
        return model.predict(X)
    
    def save_model(self, model_id: str, filepath: str):
        """Save model and metadata"""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        save_data = {
            'model_info': self.models[model_id],
            'results': self.results.get(model_id, {})
        }
        
        joblib.dump(save_data, filepath)
        
    def generate_model_report(self, model_id: str) -> str:
        """Generate comprehensive model report"""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        results = self.results[model_id]
        
        report = f"""
Machine Learning Model Report
============================

Model ID: {model_id}
Model Type: {model_info['model_type']}
Task: {model_info['task']}
Outcome: {model_info['outcome']}
Number of Features: {len(model_info['feature_names'])}

Performance Metrics
------------------
"""
        
        # Add performance metrics
        if model_info['task'] == 'classification':
            report += f"""
Training Performance:
- Accuracy: {results['train_scores']['accuracy']:.3f}
- Precision: {results['train_scores']['precision']:.3f}
- Recall: {results['train_scores']['recall']:.3f}
- F1 Score: {results['train_scores']['f1']:.3f}
"""
            if 'auc_roc' in results['train_scores']:
                report += f"- AUC-ROC: {results['train_scores']['auc_roc']:.3f}\n"
                report += f"- AUC-PR: {results['train_scores']['auc_pr']:.3f}\n"
            
            report += f"""
Test Performance:
- Accuracy: {results['test_scores']['accuracy']:.3f}
- Precision: {results['test_scores']['precision']:.3f}
- Recall: {results['test_scores']['recall']:.3f}
- F1 Score: {results['test_scores']['f1']:.3f}
"""
            if 'auc_roc' in results['test_scores']:
                report += f"- AUC-ROC: {results['test_scores']['auc_roc']:.3f}\n"
                report += f"- AUC-PR: {results['test_scores']['auc_pr']:.3f}\n"
        
        else:  # Regression
            report += f"""
Training Performance:
- RMSE: {results['train_scores']['rmse']:.3f}
- MAE: {results['train_scores']['mae']:.3f}
- R²: {results['train_scores']['r2']:.3f}
- Explained Variance: {results['train_scores']['explained_variance']:.3f}

Test Performance:
- RMSE: {results['test_scores']['rmse']:.3f}
- MAE: {results['test_scores']['mae']:.3f}
- R²: {results['test_scores']['r2']:.3f}
- Explained Variance: {results['test_scores']['explained_variance']:.3f}
"""
        
        return report