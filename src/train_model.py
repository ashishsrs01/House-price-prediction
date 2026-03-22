from data_preprocessing import preprocess_data
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

print("=" * 60)
print("HOUSE PRICE PREDICTION - MODEL TRAINING")
print("=" * 60)

# Step 1: Preprocess the data
print("\n--- STEP 1: DATA PREPROCESSING ---")
df = preprocess_data()

# Step 2: Prepare features and target variable
print("\n--- STEP 2: PREPARING FEATURES AND TARGET ---")
x = df.drop('SalePrice', axis=1)  # Features
y = df['SalePrice']  # Target variable

print(f"Features shape: {x.shape}")
print(f"Target shape: {y.shape}")

# Step 3: Split the data into training and testing sets
print("\n--- STEP 3: SPLITTING DATA ---")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)
print(f"Training set size: {len(x_train)}")
print(f"Testing set size: {len(x_test)}")

# Dictionary to store model evaluations
model_evaluations = {}

def evaluate_model(name, model, x_train, y_train, x_test, y_test):
    """Train model and evaluate it with various metrics"""
    
    print(f"\n--- Training {name} ---")
    # Train the model
    model.fit(x_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(x_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate accuracy as percentage of predictions within 15% of actual value
    tolerance = 0.15 * y_test.values
    accuracy_count = np.sum(np.abs(y_test.values - y_pred) <= tolerance)
    accuracy_percentage = (accuracy_count / len(y_test)) * 100
    
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"Accuracy (within 15%): {accuracy_percentage:.2f}%")
    
    # Store performance for comparison
    model_evaluations[name] = {
        'model': model, 
        'rmse': rmse, 
        'mae': mae, 
        'r2': r2,
        'accuracy': accuracy_percentage,
        'predictions': y_pred
    }

# Step 4: Train and evaluate models
print("\n" + "=" * 60)
print("TRAINING AND EVALUATING MODELS")
print("=" * 60)

# 1. Linear Regression
evaluate_model("Linear Regression", LinearRegression(), x_train, y_train, x_test, y_test)

# 2. Random Forest Regressor
evaluate_model("Random Forest Regressor", RandomForestRegressor(random_state=42, n_estimators=100), x_train, y_train, x_test, y_test)

# 3. Gradient Boosting Regressor
evaluate_model("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42, n_estimators=100), x_train, y_train, x_test, y_test)

# Step 5: Compare models and find the best one
print("\n" + "=" * 60)
print("MODEL COMPARISON AND RESULTS")
print("=" * 60)

best_model_name = min(model_evaluations, key=lambda k: model_evaluations[k]['rmse'])
best_model = model_evaluations[best_model_name]['model']

print("\n--- Overall Model Comparison ---")
print(f"{'Model Name':<30} {'R² Score':<12} {'RMSE':<15} {'MAE':<15} {'Accuracy':<12}")
print("-" * 85)
for name, metrics in model_evaluations.items():
    print(f"{name:<30} {metrics['r2']:<12.4f} ${metrics['rmse']:<14,.2f} ${metrics['mae']:<14,.2f} {metrics['accuracy']:<11.2f}%")

print("\n" + "=" * 60)
print(f"BEST MODEL: {best_model_name}")
print(f"R² Score: {model_evaluations[best_model_name]['r2']:.4f}")
print(f"RMSE: ${model_evaluations[best_model_name]['rmse']:,.2f}")
print(f"MAE: ${model_evaluations[best_model_name]['mae']:,.2f}")
print(f"Accuracy: {model_evaluations[best_model_name]['accuracy']:.2f}%")
print("=" * 60)

# Step 6: Show some sample predictions
print("\n--- Sample Predictions vs Actual ---")
y_pred = model_evaluations[best_model_name]['predictions']
for i in range(min(5, len(y_test))):
    print(f"Actual: ${y_test.values[i]:,.2f}  |  Predicted: ${y_pred[i]:,.2f}  |  Difference: ${abs(y_test.values[i] - y_pred[i]):,.2f}")

print("\n✓ Model training and evaluation completed successfully!")
