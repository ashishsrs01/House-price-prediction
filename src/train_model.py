from data_preprocessing import df
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

# Define features and target variables 
x = df.drop('SalePrice', axis=1)  # Features
y = df['SalePrice']  # Target variable

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# Dictionary to store model evaluations
model_evaluations = {}

def evaluate_model(name, model, x_train, y_train, x_test, y_test):
    # Train the model
    model.fit(x_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(x_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"--- {name} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}\n")
    
    # Store performance for comparison (using RMSE, lower is better)
    model_evaluations[name] = {'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2}

# Train and evaluate models
print("Training and evaluating models...\n")

# 1. Linear Regression
evaluate_model("Linear Regression", LinearRegression(), x_train, y_train, x_test, y_test)

# 2. Random Forest Regressor
evaluate_model("Random Forest Regressor", RandomForestRegressor(random_state=42), x_train, y_train, x_test, y_test)

# 3. Gradient Boosting Regressor
evaluate_model("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42), x_train, y_train, x_test, y_test)

# Compare models and find the best one
best_model_name = min(model_evaluations, key=lambda k: model_evaluations[k]['rmse'])
best_model = model_evaluations[best_model_name]['model']

print("--- Model Comparison ---")
for name, metrics in model_evaluations.items():
    print(f"{name}: RMSE = {metrics['rmse']:.4f}, R^2 = {metrics['r2']:.4f}")

print(f"\nBest Model: {best_model_name} with RMSE of {model_evaluations[best_model_name]['rmse']:.4f}")
