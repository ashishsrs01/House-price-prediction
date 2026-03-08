import data_preprocessing
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

# Assuming 'df' is the preprocessed DataFrame from data_preprocessing.py

def train_model():

    df = data_preprocessing.df  # Access the preprocessed DataFrame

    #define features and target variables 
    x = df.drop('SalesPrice', axis=1)  # Features
    y = df['SalesPrice']  # Target variable

