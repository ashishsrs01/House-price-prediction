# House Price Prediction using Machine Learning

## Project Overview

This project builds a machine learning model to predict house prices using real-world housing data.
The goal is to analyze important property features and train regression models capable of estimating house prices.

The project demonstrates a complete machine learning workflow including data preprocessing, feature engineering, model training, and model evaluation using Python.

## Problem Statement

House prices depend on multiple factors such as size, quality, location, number of rooms, and construction year.
The objective of this project is to train machine learning models that can learn patterns from historical housing data and predict the selling price of houses.

## Dataset

The dataset used in this project is the **Ames Housing Dataset**, commonly used for regression problems in machine learning.

It contains detailed information about residential homes including:

* Lot Area
* Overall Quality
* Year Built
* Number of Rooms
* Garage Size
* Basement Area
* Neighborhood

Target variable:

* **SalePrice** – the selling price of the house.

Dataset files:

```
train.csv
test.csv
```

## Project Structure

```
house-price-prediction
│
├── data
│   ├── train.csv
│   └── test.csv
│
├── src
│   ├── data_preprocessing.py
│   └── train_model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Machine Learning Workflow

The project follows a structured machine learning pipeline:

1. Load the dataset
2. Clean and preprocess the data
3. Handle missing values
4. Encode categorical features
5. Split data into training and testing sets
6. Train regression models
7. Evaluate model performance

## Algorithms Used

The following regression algorithms are implemented and compared:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

These models are evaluated to determine which performs best for predicting house prices.

## Model Evaluation

Model performance is evaluated using standard regression metrics:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

These metrics measure how accurately the model predicts house prices.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/yourusername/house-price-prediction.git
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the training script

```
python src/train_model.py
```

The script will train the models and print the evaluation results.

## Future Improvements

* Hyperparameter tuning for better performance
* Feature importance analysis
* Model deployment as a web application
* Integration with a prediction interface

## Author

Ashish Sharma

AI & Data Science Student at IIT jodhpur, passionate about Machine Learning, Artificial Intelligence, and real-world data analysis.

## License

This project is open-source and available under the MIT License.
