"""
Model evaluation module for Student Performance Prediction project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           accuracy_score, precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import learning_curve, validation_curve
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Class for evaluating machine learning models on student performance data.
    """
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_regression_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate regression model performance.
        
        Parameters:
        model: Trained model
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): True test targets
        model_name (str): Name of the model
        
        Returns:
        dict: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        mean_actual = np.mean(y_test)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
        
        results = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'MAPE': mape,
            'Mean_Actual': mean_actual,
            'Predictions': y_pred
        }
        
        self.evaluation_results[model_name] = results
        
        print(f"Regression Evaluation Results for {model_name}:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return results
    
    def evaluate_classification_model(self, model, X_test, y_test, model_name="Model", average='weighted'):
        """
        Evaluate classification model performance.
        
        Parameters:
        model: Trained model
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): True test targets
        model_name (str): Name of the model
        average (str): Averaging method for multi-class metrics
        
        Returns:
        dict: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
        
        # Get classification report
        class_report = classification_report(y_test, y_pred)
        
        # Calculate AUC-ROC for binary classification
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:  # Binary classification
                    auc_roc = roc_auc_score(y_test, y_proba[:, 1])
                else:  # Multi-class
                    auc_roc = roc_auc_score(y_test, y_proba, multi_class='ovr', average=average)
            else:
                auc_roc = None
        except:
            auc_roc = None
        
        results = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'AUC_ROC': auc_roc,
            'Classification_Report': class_report,
            'Predictions': y_pred
        }
        
        self.evaluation_results[model_name] = results
        
        print(f"Classification Evaluation Results for {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if auc_roc:
            print(f"AUC-ROC: {auc_roc:.4f}")
        print("\nClassification Report:")
        print(class_report)
        
        return results
    
    def compare_models(self, models_results, problem_type='regression'):
        """
        Compare multiple models performance.
        
        Parameters:
        models_results (dict): Dictionary of model evaluation results
        problem_type (str): 'regression' or 'classification'
        
        Returns:
        pandas.DataFrame: Comparison table
        """
        if problem_type == 'regression':
            metrics = ['MSE', 'RMSE', 'MAE', 'R2_Score', 'MAPE']
        else:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC']
        
        comparison_data = {}
        for model_name, results in models_results.items():
            comparison_data[model_name] = [results.get(metric, np.nan) for metric in metrics]
        
        comparison_df = pd.DataFrame(comparison_data, index=metrics).T
        
        print(f"\nModel Comparison ({problem_type.capitalize()}):")
        print("=" * 60)
        print(comparison_df.round(4))
        
        # Highlight best performing models
        if problem_type == 'regression':
            print(f"\nBest Models:")
            print(f"Lowest RMSE: {comparison_df['RMSE'].idxmin()} ({comparison_df['RMSE'].min():.4f})")
            print(f"Highest R²: {comparison_df['R2_Score'].idxmax()} ({comparison_df['R2_Score'].max():.4f})")
        else:
            print(f"\nBest Models:")
            print(f"Highest Accuracy: {comparison_df['Accuracy'].idxmax()} ({comparison_df['Accuracy'].max():.4f})")
            print(f"Highest F1 Score: {comparison_df['F1_Score'].idxmax()} ({comparison_df['F1_Score'].max():.4f})")
        
        return comparison_df
    
    def plot_predictions_vs_actual(self, y_test, y_pred, model_name="Model"):
        """
        Plot predictions vs actual values for regression.
        
        Parameters:
        y_test (pandas.Series): True test targets
        y_pred (numpy.array): Predicted values
        model_name (str): Name of the model
        """
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}: Predicted vs Actual')
        
        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name}: Residual Plot')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name="Model", labels=None):
        """
        Plot confusion matrix for classification.
        
        Parameters:
        y_test (pandas.Series): True test targets
        y_pred (numpy.array): Predicted values
        model_name (str): Name of the model
        labels (list): Class labels
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'{model_name}: Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
    
    def plot_roc_curve(self, model, X_test, y_test, model_name="Model"):
        """
        Plot ROC curve for binary classification.
        
        Parameters:
        model: Trained model
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): True test targets
        model_name (str): Name of the model
        """
        if not hasattr(model, "predict_proba"):
            print(f"Model {model_name} doesn't support probability prediction.")
            return
        
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name}: ROC Curve')
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Error plotting ROC curve: {e}")
    
    def plot_learning_curves(self, model, X, y, model_name="Model", cv=5):
        """
        Plot learning curves to analyze model performance vs training size.
        
        Parameters:
        model: Model to evaluate
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        model_name (str): Name of the model
        cv (int): Cross-validation folds
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(f'{model_name}: Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def generate_evaluation_report(self, models_results, problem_type='regression'):
        """
        Generate a comprehensive evaluation report.
        
        Parameters:
        models_results (dict): Dictionary of model evaluation results
        problem_type (str): 'regression' or 'classification'
        
        Returns:
        str: Formatted evaluation report
        """
        report = f"\n{'='*60}\n"
        report += f"MODEL EVALUATION REPORT - {problem_type.upper()}\n"
        report += f"{'='*60}\n\n"
        
        # Model comparison
        comparison_df = self.compare_models(models_results, problem_type)
        report += f"Model Performance Comparison:\n"
        report += f"{comparison_df.to_string()}\n\n"
        
        # Best model identification
        if problem_type == 'regression':
            best_model_rmse = comparison_df['RMSE'].idxmin()
            best_model_r2 = comparison_df['R2_Score'].idxmax()
            report += f"Best Model (Lowest RMSE): {best_model_rmse}\n"
            report += f"Best Model (Highest R²): {best_model_r2}\n\n"
        else:
            best_model_acc = comparison_df['Accuracy'].idxmax()
            best_model_f1 = comparison_df['F1_Score'].idxmax()
            report += f"Best Model (Highest Accuracy): {best_model_acc}\n"
            report += f"Best Model (Highest F1 Score): {best_model_f1}\n\n"
        
        # Recommendations
        report += "RECOMMENDATIONS:\n"
        report += "-" * 20 + "\n"
        
        if problem_type == 'regression':
            rmse_values = comparison_df['RMSE'].dropna()
            if not rmse_values.empty:
                best_rmse = rmse_values.min()
                worst_rmse = rmse_values.max()
                report += f"• The {best_model_rmse} model shows the best predictive performance\n"
                report += f"• RMSE ranges from {best_rmse:.4f} to {worst_rmse:.4f}\n"
                
                if best_rmse < 10:
                    report += f"• Excellent model performance (RMSE < 10)\n"
                elif best_rmse < 15:
                    report += f"• Good model performance (RMSE < 15)\n"
                else:
                    report += f"• Consider feature engineering or more complex models\n"
        else:
            acc_values = comparison_df['Accuracy'].dropna()
            if not acc_values.empty:
                best_acc = acc_values.max()
                report += f"• The {best_model_acc} model shows the best accuracy\n"
                
                if best_acc > 0.9:
                    report += f"• Excellent model performance (Accuracy > 90%)\n"
                elif best_acc > 0.8:
                    report += f"• Good model performance (Accuracy > 80%)\n"
                else:
                    report += f"• Consider feature engineering or more complex models\n"
        
        return report

def main():
    """
    Example usage of the ModelEvaluator class.
    """
    # Create sample data for demonstration
    np.random.seed(42)
    
    # Generate synthetic test data
    n_test = 200
    X_test = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_test),
        'feature2': np.random.normal(0, 1, n_test),
        'feature3': np.random.normal(0, 1, n_test)
    })
    
    # Generate regression targets
    y_test_reg = 50 + X_test['feature1'] * 10 + X_test['feature2'] * 5 + np.random.normal(0, 5, n_test)
    
    # Generate classification targets
    y_test_class = (y_test_reg > y_test_reg.median()).astype(int)
    
    # Create mock predictions (normally you'd get these from trained models)
    y_pred_reg_1 = y_test_reg + np.random.normal(0, 3, n_test)  # Good model
    y_pred_reg_2 = y_test_reg + np.random.normal(0, 8, n_test)  # Worse model
    
    y_pred_class_1 = (y_pred_reg_1 > y_pred_reg_1.median()).astype(int)  # Good model
    
    print("Sample data created for demonstration")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate regression models
    print("\n" + "="*50)
    print("REGRESSION EVALUATION")
    print("="*50)
    
    # Mock regression model evaluations
    reg_results = {}
    reg_results['Model_A'] = {
        'MSE': mean_squared_error(y_test_reg, y_pred_reg_1),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg_1)),
        'MAE': mean_absolute_error(y_test_reg, y_pred_reg_1),
        'R2_Score': r2_score(y_test_reg, y_pred_reg_1),
        'MAPE': np.mean(np.abs((y_test_reg - y_pred_reg_1) / y_test_reg)) * 100
    }
    
    reg_results['Model_B'] = {
        'MSE': mean_squared_error(y_test_reg, y_pred_reg_2),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg_2)),
        'MAE': mean_absolute_error(y_test_reg, y_pred_reg_2),
        'R2_Score': r2_score(y_test_reg, y_pred_reg_2),
        'MAPE': np.mean(np.abs((y_test_reg - y_pred_reg_2) / y_test_reg)) * 100
    }
    
    # Compare regression models
    comparison_df = evaluator.compare_models(reg_results, 'regression')
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(reg_results, 'regression')
    print(report)
    
    print("Model evaluation completed successfully!")

if __name__ == "__main__":
    main()