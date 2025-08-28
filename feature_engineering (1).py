"""
Feature engineering module for Student Performance Prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Class for feature engineering and selection in student performance prediction.
    """
    
    def __init__(self):
        self.selected_features = None
        self.feature_selector = None
        self.poly_features = None
        
    def create_interaction_features(self, df, categorical_cols=None, numerical_cols=None):
        """
        Create interaction features between categorical and numerical variables.
        
        Parameters:
        df (pandas.DataFrame): Input dataframe
        categorical_cols (list): List of categorical columns
        numerical_cols (list): List of numerical columns
        
        Returns:
        pandas.DataFrame: Dataframe with interaction features
        """
        df_interactions = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print("Creating interaction features...")
        
        # Create interactions between categorical variables
        if len(categorical_cols) > 1:
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    interaction_name = f"{col1}_x_{col2}"
                    df_interactions[interaction_name] = df_interactions[col1].astype(str) + "_" + df_interactions[col2].astype(str)
                    print(f"Created interaction: {interaction_name}")
        
        return df_interactions
    
    def create_polynomial_features(self, df, numerical_cols=None, degree=2, include_bias=False):
        """
        Create polynomial features from numerical columns.
        
        Parameters:
        df (pandas.DataFrame): Input dataframe
        numerical_cols (list): List of numerical columns
        degree (int): Degree of polynomial features
        include_bias (bool): Whether to include bias term
        
        Returns:
        pandas.DataFrame: Dataframe with polynomial features
        """
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_poly = df.copy()
        
        if len(numerical_cols) > 0:
            self.poly_features = PolynomialFeatures(degree=degree, include_bias=include_bias)
            
            # Fit and transform numerical features
            numerical_data = df[numerical_cols]
            poly_data = self.poly_features.fit_transform(numerical_data)
            
            # Get feature names
            poly_feature_names = self.poly_features.get_feature_names_out(numerical_cols)
            
            # Create polynomial features dataframe
            poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=df.index)
            
            # Remove original features to avoid duplication
            for col in numerical_cols:
                if col in poly_df.columns:
                    poly_df = poly_df.drop(col, axis=1)
            
            # Combine with original dataframe
            df_poly = pd.concat([df_poly, poly_df], axis=1)
            
            print(f"Created {len(poly_feature_names) - len(numerical_cols)} polynomial features")
        
        return df_poly
    
    def create_binned_features(self, df, numerical_cols=None, n_bins=5, strategy='quantile'):
        """
        Create binned versions of numerical features.
        
        Parameters:
        df (pandas.DataFrame): Input dataframe
        numerical_cols (list): List of numerical columns to bin
        n_bins (int): Number of bins
        strategy (str): Binning strategy ('uniform', 'quantile', 'kmeans')
        
        Returns:
        pandas.DataFrame: Dataframe with binned features
        """
        df_binned = df.copy()
        
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Creating binned features with {n_bins} bins using {strategy} strategy...")
        
        for col in numerical_cols:
            if col in df_binned.columns:
                try:
                    if strategy == 'quantile':
                        df_binned[f"{col}_binned"] = pd.cut(df_binned[col], bins=n_bins, labels=False)
                    elif strategy == 'uniform':
                        df_binned[f"{col}_binned"] = pd.cut(df_binned[col], bins=n_bins, labels=False)
                    elif strategy == 'kmeans':
                        from sklearn.cluster import KMeans
                        kmeans = KMeans(n_clusters=n_bins, random_state=42)
                        df_binned[f"{col}_binned"] = kmeans.fit_predict(df_binned[[col]])
                    
                    print(f"Created binned feature: {col}_binned")
                except Exception as e:
                    print(f"Error creating binned feature for {col}: {e}")
        
        return df_binned
    
    def create_score_ratios(self, df):
        """
        Create ratio features between different scores.
        
        Parameters:
        df (pandas.DataFrame): Input dataframe
        
        Returns:
        pandas.DataFrame: Dataframe with ratio features
        """
        df_ratios = df.copy()
        
        score_columns = [col for col in df.columns if 'score' in col.lower()]
        
        if len(score_columns) > 1:
            print("Creating score ratio features...")
            
            for i, col1 in enumerate(score_columns):
                for col2 in score_columns[i+1:]:
                    # Avoid division by zero
                    if (df_ratios[col2] != 0).all():
                        ratio_name = f"{col1}_to_{col2}_ratio"
                        df_ratios[ratio_name] = df_ratios[col1] / df_ratios[col2]
                        print(f"Created ratio feature: {ratio_name}")
        
        return df_ratios
    
    def create_performance_categories(self, df, score_columns=None):
        """
        Create performance category features based on score thresholds.
        
        Parameters:
        df (pandas.DataFrame): Input dataframe
        score_columns (list): List of score columns
        
        Returns:
        pandas.DataFrame: Dataframe with performance categories
        """
        df_categories = df.copy()
        
        if score_columns is None:
            score_columns = [col for col in df.columns if 'score' in col.lower()]
        
        print("Creating performance category features...")
        
        for col in score_columns:
            if col in df_categories.columns:
                # Create performance categories
                df_categories[f"{col}_category"] = pd.cut(
                    df_categories[col],
                    bins=[0, 50, 70, 85, 100],
                    labels=['Poor', 'Average', 'Good', 'Excellent'],
                    include_lowest=True
                )
                
                # Create binary high/low performance
                df_categories[f"{col}_high_performer"] = (df_categories[col] >= df_categories[col].median()).astype(int)
                
                print(f"Created category features for {col}")
        
        return df_categories
    
    def select_features_univariate(self, X, y, k=10, problem_type='regression'):
        """
        Select features using univariate statistical tests.
        
        Parameters:
        X (pandas.DataFrame): Feature matrix
        y (pandas.Series): Target vector
        k (int): Number of features to select
        problem_type (str): 'regression' or 'classification'
        
        Returns:
        tuple: (selected_features, feature_selector)
        """
        if problem_type == 'regression':
            score_func = f_regression
        else:
            score_func = f_classif
        
        self.feature_selector = SelectKBest(score_func=score_func, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        print(f"Selected {len(self.selected_features)} features using univariate selection:")
        for i, feature in enumerate(self.selected_features):
            score = self.feature_selector.scores_[selected_mask][i]
            print(f"  {feature}: {score:.4f}")
        
        return self.selected_features, self.feature_selector
    
    def select_features_rfe(self, X, y, n_features=10, problem_type='regression'):
        """
        Select features using Recursive Feature Elimination.
        
        Parameters:
        X (pandas.DataFrame): Feature matrix
        y (pandas.Series): Target vector
        n_features (int): Number of features to select
        problem_type (str): 'regression' or 'classification'
        
        Returns:
        tuple: (selected_features, feature_selector)
        """
        if problem_type == 'regression':
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.feature_selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        print(f"Selected {len(self.selected_features)} features using RFE:")
        for i, feature in enumerate(self.selected_features):
            ranking = self.feature_selector.ranking_[X.columns.get_loc(feature)]
            print(f"  {feature}: rank {ranking}")
        
        return self.selected_features, self.feature_selector
    
    def select_features_importance(self, X, y, n_features=10, problem_type='regression', threshold=None):
        """
        Select features based on feature importance from Random Forest.
        
        Parameters:
        X (pandas.DataFrame): Feature matrix
        y (pandas.Series): Target vector
        n_features (int): Number of features to select (if threshold is None)
        problem_type (str): 'regression' or 'classification'
        threshold (float): Importance threshold (if provided, overrides n_features)
        
        Returns:
        tuple: (selected_features, feature_importances)
        """
        if problem_type == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        
        # Get feature importances
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select features based on threshold or top n_features
        if threshold is not None:
            self.selected_features = feature_importances[
                feature_importances['importance'] >= threshold
            ]['feature'].tolist()
        else:
            self.selected_features = feature_importances.head(n_features)['feature'].tolist()
        
        print(f"Selected {len(self.selected_features)} features based on importance:")
        selected_importances = feature_importances[
            feature_importances['feature'].isin(self.selected_features)
        ]
        
        for _, row in selected_importances.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.selected_features, feature_importances
    
    def apply_feature_selection(self, X, feature_names=None):
        """
        Apply previously fitted feature selection to new data.
        
        Parameters:
        X (pandas.DataFrame): Feature matrix
        feature_names (list): List of feature names to select (overrides fitted selector)
        
        Returns:
        pandas.DataFrame: Selected features
        """
        if feature_names is not None:
            return X[feature_names]
        elif self.selected_features is not None:
            return X[self.selected_features]
        elif self.feature_selector is not None:
            return pd.DataFrame(
                self.feature_selector.transform(X),
                columns=self.selected_features,
                index=X.index
            )
        else:
            print("No feature selection method has been fitted.")
            return X
    
    def get_feature_engineering_report(self, original_features, final_features):
        """
        Generate a report of feature engineering steps.
        
        Parameters:
        original_features (list): List of original feature names
        final_features (list): List of final feature names
        
        Returns:
        str: Feature engineering report
        """
        report = "\n" + "="*60 + "\n"
        report += "FEATURE ENGINEERING REPORT\n"
        report += "="*60 + "\n\n"
        
        report += f"Original number of features: {len(original_features)}\n"
        report += f"Final number of features: {len(final_features)}\n"
        report += f"Features added: {len(final_features) - len(original_features)}\n\n"
        
        # New features created
        new_features = [f for f in final_features if f not in original_features]
        if new_features:
            report += f"New features created ({len(new_features)}):\n"
            for feature in new_features[:10]:  # Show first 10
                report += f"  - {feature}\n"
            if len(new_features) > 10:
                report += f"  ... and {len(new_features) - 10} more\n"
            report += "\n"
        
        # Selected features
        if self.selected_features:
            report += f"Selected features ({len(self.selected_features)}):\n"
            for feature in self.selected_features[:10]:  # Show first 10
                report += f"  - {feature}\n"
            if len(self.selected_features) > 10:
                report += f"  ... and {len(self.selected_features) - 10} more\n"
        
        return report

def main():
    """
    Example usage of the FeatureEngineer class.
    """
    # Create sample student performance data
    np.random.seed(42)
    
    sample_data = {
        'gender': np.random.choice(['male', 'female'], 1000),
        'race_ethnicity': np.random.choice(['group A', 'group B', 'group C', 'group D', 'group E'], 1000),
        'parental_education': np.random.choice(['high school', 'some college', "bachelor's degree", "master's degree"], 1000),
        'lunch': np.random.choice(['standard', 'free/reduced'], 1000),
        'test_preparation': np.random.choice(['completed', 'none'], 1000),
        'math_score': np.random.normal(70, 15, 1000),
        'reading_score': np.random.normal(70, 15, 1000),
        'writing_score': np.random.normal(70, 15, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Clip scores to realistic range
    score_cols = ['math_score', 'reading_score', 'writing_score']
    for col in score_cols:
        df[col] = np.clip(df[col], 0, 100)
    
    print("Sample data created for demonstration")
    print(f"Original shape: {df.shape}")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Create interaction features
    df_interactions = engineer.create_interaction_features(
        df, 
        categorical_cols=['gender', 'lunch', 'test_preparation'],
        numerical_cols=['reading_score', 'writing_score']
    )
    
    # Create score ratios
    df_ratios = engineer.create_score_ratios(df_interactions)
    
    # Create performance categories
    df_categories = engineer.create_performance_categories(df_ratios)
    
    # Create binned features
    df_binned = engineer.create_binned_features(
        df_categories, 
        numerical_cols=['reading_score', 'writing_score']
    )
    
    print(f"After feature engineering: {df_binned.shape}")
    
    # Prepare for feature selection (encode categorical variables)
    df_encoded = pd.get_dummies(df_binned, drop_first=True)
    
    # Separate features and target
    X = df_encoded.drop(['math_score'], axis=1)
    y = df_encoded['math_score']
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Feature selection using importance
    selected_features, importance_df = engineer.select_features_importance(
        X, y, n_features=15, problem_type='regression'
    )
    
    # Apply feature selection
    X_selected = engineer.apply_feature_selection(X)
    print(f"Selected features shape: {X_selected.shape}")
    
    # Generate report
    report = engineer.get_feature_engineering_report(
        df.columns.tolist(), 
        X_selected.columns.tolist()
    )
    print(report)
    
    print("Feature engineering completed successfully!")

if __name__ == "__main__":
    main()