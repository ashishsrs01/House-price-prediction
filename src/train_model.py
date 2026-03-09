from data_preprocessing import df
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

# Assuming 'df' is the preprocessed DataFrame from data_preprocessing.py



#define features and target variables 
x = df.drop('SalePrice', axis=1)  # Features
y = df['SalePrice']  # Target variable

#Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 30)


# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

    #Predict on the test set from LR model
y_pred_lr = lr_model.predict(x_test)

    # Evaluate Lr_Model 
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)
print(f"Mean Squared Error: {mse_lr}")
print(f"R^2 Score: {r2}")
    
    # Train Random Forest Regression Model
rf_model = RandomForestRegressor()
rf_model.fit(x_train, y_train)

    # Predict on the test set from RF model
y_pred_rf = rf_model.predict(x_test)

    # Evaluate Rf_Model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Mean Squared Error: {mse_rf}")  
print(f"R^2 Score: {r2_rf}")

