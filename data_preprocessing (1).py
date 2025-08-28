"""
Data preprocessing module for Student Performance Prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Class for preprocessing student performance data.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(drop='first', sparse=False)
        
    def load_data(self, filepath):
        """
        Load data from CSV file.
        
        Parameters:
        filepath (str): Path to the CSV file
        
        Returns:
        pandas.DataFrame: Loaded dataframe
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self, df):
        """
        Basic data exploration.
        
        Parameters:
        df (pandas.DataFrame): Input dataframe
        
        Returns:
        dict: Dictionary containing basic statistics
        """
        exploration_results = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'null_values': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe(),
            'categorical_summary': df.describe(include=['object'])
        }
        
        return exploration_results
    
    def clean_data(self, df):
        """
        Clean the dataset.
        
        Parameters:
        df (pandas.DataFrame): Input dataframe
        
        Returns:
        pandas.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Remove duplicates
        initial_shape = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_shape - df_clean.shape[0]
        if removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values if any
        if df_clean.isnull().sum().sum() > 0:
            print("Missing values found:")
            print(df_clean.isnull().sum())
            # Fill numeric columns with median
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        return df_clean
    
    def encode_categorical_features(self, df, method='onehot'):
        """
        Encode categorical features.
        
        Parameters:
        df (pandas.DataFrame): Input dataframe
        method (str): Encoding method - 'onehot' or 'label'
        
        Returns:
        pandas.DataFrame: Dataframe with encoded features
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        if method == 'onehot':
            # One-hot encoding
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
        
        elif method == 'label':
            # Label encoding
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
        
        return df_encoded
    
    def create_features(self, df):
        """
        Create additional features.
        
        Parameters:
        df (pandas.DataFrame): Input dataframe
        
        Returns:
        pandas.DataFrame: Dataframe with new features
        """
        df_features = df.copy()
        
        # Create total score if reading and writing scores are available
        if 'reading score' in df_features.columns and 'writing score' in df_features.columns:
            df_features['total_score'] = df_features['reading score'] + df_features['writing score']
            if 'math score' in df_features.columns:
                df_features['average_score'] = (df_features['math score'] + 
                                               df_features['reading score'] + 
                                               df_features['writing score']) / 3
        
        # Create score categories
        if 'math score' in df_features.columns:
            df_features['math_score_category'] = pd.cut(df_features['math score'], 
                                                       bins=[0, 50, 70, 85, 100], 
                                                       labels=['Poor', 'Average', 'Good', 'Excellent'])
        
        return df_features
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale numerical features.
        
        Parameters:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Test features (optional)
        
        Returns:
        tuple: Scaled training and test features
        """
        # Identify numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # Scale training data
        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        
        # Scale test data if provided
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_data_for_modeling(self, df, target_col='math score', test_size=0.2, random_state=42):
        """
        Prepare data for machine learning modeling.
        
        Parameters:
        df (pandas.DataFrame): Input dataframe
        target_col (str): Target column name
        test_size (float): Test set size
        random_state (int): Random state for reproducibility
        
        Returns:
        tuple: X_train, X_test, y_train, y_test
        """
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Target distribution in training set:")
        print(y_train.describe())
        
        return X_train, X_test, y_train, y_test

def main():
    """
    Example usage of the DataPreprocessor class.
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data (uncomment when you have the dataset)
    # df = preprocessor.load_data('data/raw/StudentsPerformance.csv')
    
    # Example with sample data
    sample_data = {
        'gender': ['female', 'male', 'female', 'male', 'female'],
        'race/ethnicity': ['group A', 'group B', 'group A', 'group C', 'group B'],
        'parental level of education': ['some college', 'high school', 'bachelor\'s degree', 'some college', 'master\'s degree'],
        'lunch': ['standard', 'free/reduced', 'standard', 'standard', 'free/reduced'],
        'test preparation course': ['completed', 'none', 'completed', 'none', 'completed'],
        'math score': [67, 45, 78, 56, 89],
        'reading score': [72, 51, 83, 61, 92],
        'writing score': [70, 48, 80, 58, 88]
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample data created for demonstration")
    
    # Explore data
    exploration = preprocessor.explore_data(df)
    print(f"\nData shape: {exploration['shape']}")
    print(f"Columns: {exploration['columns']}")
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Create features
    df_features = preprocessor.create_features(df_clean)
    
    # Encode categorical features
    df_encoded = preprocessor.encode_categorical_features(df_features, method='onehot')
    
    # Prepare for modeling
    X_train, X_test, y_train, y_test = preprocessor.prepare_data_for_modeling(df_encoded)
    
    print("\nData preprocessing completed successfully!")

if __name__ == "__main__":
    main()