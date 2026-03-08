import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Load Dataset
Data = pd.read_csv("Data/train.csv")
if Data.isnull().sum().any():
    print("Missing values found in the dataset. Please handle them before proceeding.")
else: 
    print("No missing values found in the dataset. Proceeding with preprocessing.")

# Display basic information about the dataset
print("Dataset Shape:", Data.shape)

print("\nColumn Information:")
print(Data.info())

print("\nFirst 5 Rows:")
print(Data.head())


# Data cleaning and preprocessing

# Identify numerical and categorical columns
numerical_cols = Data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = Data.select_dtypes(include=['object']).columns

# checking missing values in numerical and categorical columns
print("\nMissing values in numerical columns:")
print(Data[numerical_cols].isnull().sum())
print("\nMissing values in categorical columns:")
print(Data[categorical_cols].isnull().sum())

# Handle missing values (if any)

# For numerical columns, we can fill missing values with the mean
Data[numerical_cols] = Data[numerical_cols].fillna(Data[numerical_cols].mean())

# For categorical columns, we can fill missing values with the mode 
Data[categorical_cols] = Data[categorical_cols].fillna(Data[categorical_cols].mode().iloc[0])

# Encoding categorical variables using OneHotEncoder
df = pd.get_dummies(Data, columns=categorical_cols)


