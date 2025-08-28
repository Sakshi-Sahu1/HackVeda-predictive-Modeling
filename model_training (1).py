"""
Model training module for Student Performance Prediction project.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Class for training machine learning models on student performance data.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.model_scores = {}
    
    def get_regression_models(self):
        """
        Get regression models for continuous target prediction.
        
        Returns:
        dict: Dictionary of regression models
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'Support Vector Regression': SVR()
        }
        return models
    
    def get_classification_models(self):
        """
        Get classification models for categorical target prediction.
        
        Returns:
        dict: Dictionary of classification models
        """
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Support Vector Machine': SVC(random_state=42)
        }
        return models
    
    def get_hyperparameter_grids(self, model_type='regression'):
        """
        Get hyperparameter grids for model tuning.
        
        Parameters:
        model_type (str): 'regression' or 'classification'
        
        Returns:
        dict: Hyperparameter grids
        """
        if model_type == 'regression':
            param_grids = {
                'Decision Tree': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Random Forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Support Vector Regression': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'linear']
                }
            }
        else:
            param_grids = {
                'Logistic Regression': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                'Decision Tree': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Random Forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Support Vector Machine': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'linear']
                }
            }
        
        return param_grids
    
    def train_models(self, X_train, y_train, problem_type='regression', cv=5):
        """
        Train multiple models and compare their performance.
        
        Parameters:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        problem_type (str): 'regression' or 'classification'
        cv (int): Number of cross-validation folds
        
        Returns:
        dict: Trained models with their scores
        """
        if problem_type == 'regression':
            models = self.get_regression_models()
            scoring = 'neg_mean_squared_error'
        else:
            models = self.get_classification_models()
            scoring = 'accuracy'
        
        print(f"Training {len(models)} {problem_type} models...")
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Store model and scores
            self.models[name] = model
            if problem_type == 'regression':
                self.model_scores[name] = {
                    'cv_score_mean': -cv_scores.mean(),  # Convert back from negative MSE
                    'cv_score_std': cv_scores.std(),
                    'cv_rmse_mean': np.sqrt(-cv_scores.mean())
                }
            else:
                self.model_scores[name] = {
                    'cv_score_mean': cv_scores.mean(),
                    'cv_score_std': cv_scores.std()
                }
            
            print(f"CV Score: {self.model_scores[name]['cv_score_mean']:.4f} (+/- {self.model_scores[name]['cv_score_std']:.4f})")
        
        # Find best model
        if problem_type == 'regression':
            best_model_name = min(self.model_scores.keys(), 
                                key=lambda k: self.model_scores[k]['cv_score_mean'])
        else:
            best_model_name = max(self.model_scores.keys(), 
                                key=lambda k: self.model_scores[k]['cv_score_mean'])
        
        self.best_model = self.models[best_model_name]
        print(f"\nBest model: {best_model_name}")
        
        return self.models
    
    def hyperparameter_tuning(self, X_train, y_train, model_names=None, problem_type='regression'):
        """
        Perform hyperparameter tuning for selected models.
        
        Parameters:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        model_names (list): List of model names to tune
        problem_type (str): 'regression' or 'classification'
        
        Returns:
        dict: Tuned models
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        param_grids = self.get_hyperparameter_grids(problem_type)
        tuned_models = {}
        
        if problem_type == 'regression':
            scoring = 'neg_mean_squared_error'
        else:
            scoring = 'accuracy'
        
        print("Starting hyperparameter tuning...")
        
        for name in model_names:
            if name not in self.models or name not in param_grids:
                continue
                
            print(f"\nTuning {name}...")
            
            grid_search = GridSearchCV(
                self.models[name],
                param_grids[name],
                cv=5,
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            tuned_models[name] = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best score: {grid_search.best_score_:.4f}")
        
        # Update models with tuned versions
        self.models.update(tuned_models)
        
        return tuned_models
    
    def get_feature_importance(self, model_name=None, feature_names=None):
        """
        Get feature importance from tree-based models.
        
        Parameters:
        model_name (str): Name of the model
        feature_names (list): List of feature names
        
        Returns:
        pandas.DataFrame: Feature importance scores
        """
        if model_name is None:
            model = self.best_model
            model_name = "Best Model"
        else:
            model = self.models.get(model_name)
        
        if model is None:
            print(f"Model {model_name} not found.")
            return None
        
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not support feature importance.")
            return None
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_name, filepath):
        """
        Save a trained model to disk.
        
        Parameters:
        model_name (str): Name of the model to save
        filepath (str): Path to save the model
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
        
        joblib.dump(self.models[model_name], filepath)
        print(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath, model_name):
        """
        Load a model from disk.
        
        Parameters:
        filepath (str): Path to the saved model
        model_name (str): Name to assign to the loaded model
        """
        try:
            model = joblib.load(filepath)
            self.models[model_name] = model
            print(f"Model loaded from {filepath} as {model_name}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

def main():
    """
    Example usage of the ModelTrainer class.
    """
    # Create sample data for demonstration
    np.random.seed(42)
    
    # Generate synthetic student performance data
    n_samples = 1000
    X_train = pd.DataFrame({
        'gender_male': np.random.randint(0, 2, n_samples),
        'race_ethnicity_B': np.random.randint(0, 2, n_samples),
        'race_ethnicity_C': np.random.randint(0, 2, n_samples),
        'lunch_standard': np.random.randint(0, 2, n_samples),
        'test_prep_completed': np.random.randint(0, 2, n_samples),
        'parental_education_level': np.random.randint(1, 6, n_samples),
        'reading_score': np.random.normal(70, 15, n_samples),
        'writing_score': np.random.normal(70, 15, n_samples)
    })
    
    # Generate math scores with some correlation to other features
    y_train = (50 + 
              X_train['gender_male'] * 5 +
              X_train['lunch_standard'] * 10 +
              X_train['test_prep_completed'] * 15 +
              X_train['parental_education_level'] * 3 +
              X_train['reading_score'] * 0.3 +
              X_train['writing_score'] * 0.2 +
              np.random.normal(0, 10, n_samples))
    
    # Clip scores to realistic range
    y_train = np.clip(y_train, 0, 100)
    
    print("Sample data created for demonstration")
    print(f"Training data shape: {