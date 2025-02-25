# Car Price Prediction

## Overview
This project focuses on predicting car prices using machine learning models. The dataset is sourced from Kaggle and includes various features such as mileage, model, and other numerical and categorical attributes.

## Dataset
- **Source**: [Car Price Dataset on Kaggle](https://www.kaggle.com/datasets/asinow/car-price-dataset/data)
- **File**: `car_price_dataset.csv`
- **Features**:
  - `Model` (Categorical)
  - `Mileage` (Numerical)
  - `Price` (Target variable)
  - Other numerical and categorical attributes

## Libraries Used
- pandas
- seaborn
- matplotlib
- scikit-learn
- scipy
- pickle

## Data Preprocessing
- Load and inspect dataset (`.info()`, `.describe()`, `.isnull().sum()`, `.columns`)
- Remove missing values if any
- Encode categorical variables using `LabelEncoder`
- Scale numerical features using `MinMaxScaler`
- Visualize data relationships using heatmaps and regression plots

## Model Training
- Split data into training and testing sets (`train_test_split`)
- Train models:
  - **Lasso Regression** (`Lasso` from `sklearn.linear_model`)
  - **Elastic Net Regression** (`ElasticNet` from `sklearn.linear_model`)
- Save trained models using `pickle`

## Model Evaluation
- Load saved models
- Predict test data
- Calculate Mean Squared Error (MSE) for model performance
- Plot actual vs. predicted values

## How to Run
1. Clone the repository:
   ```sh
   git clone https://github.com/M-Haris72/Car_Price_Prediction_Model.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the preprocessing and training script:
   ```sh
   python creatin_model.py
   ```
4. Evaluate models:
   ```sh
   python Predicted_file.py
   ```

## Future Improvements
- Implement more regression models (Random Forest, XGBoost, etc.)
- Hyperparameter tuning for better accuracy
- Deploy the model using Flask or FastAPI

## Author
- **Muhammad Haris**
- **Email**: haris72.info@gmail.com

