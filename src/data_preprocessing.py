import pandas as pd
import numpy as np
import os

def preprocess_data():
    """Preprocess the training data and return cleaned dataframe"""
    
    # Change to the correct directory if needed
    if not os.path.exists("Data/train.csv"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(script_dir))

    print("Loading dataset...")
    # Load Dataset
    Data = pd.read_csv("Data/train.csv")

    # Display basic information about the dataset
    print("Dataset Shape:", Data.shape)
    print("\nFirst 5 Rows:")
    print(Data.head())

    # Identify numerical and categorical columns
    numerical_cols = Data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = Data.select_dtypes(include=['object']).columns.tolist()

    # Check for missing values
    total_missing = Data.isnull().sum().sum()
    if total_missing > 0:
        print(f"\nMissing values found: {total_missing}")
        print("\nMissing values in numerical columns:")
        print(Data[numerical_cols].isnull().sum())
        print("\nMissing values in categorical columns:")
        print(Data[categorical_cols].isnull().sum())
    else:
        print("\nNo missing values found in the dataset.")

    # Handle missing values
    print("\nHandling missing values...")
    # For numerical columns, fill with mean
    for col in numerical_cols:
        if Data[col].isnull().sum() > 0:
            Data[col] = Data[col].fillna(Data[col].mean())

    # For categorical columns, fill with the most common value (mode)
    for col in categorical_cols:
        if Data[col].isnull().sum() > 0:
            Data[col] = Data[col].fillna(Data[col].mode()[0])

    # Encode categorical variables using one-hot encoding
    print("Encoding categorical variables...")
    df = pd.get_dummies(Data, columns=categorical_cols, drop_first=True)

    print("\nData after preprocessing:")
    print(df.head())
    print(f"Preprocessed data shape: {df.shape}")
    print("\nPreprocessing completed successfully! Dataset is ready for modeling.")
    
    return df

