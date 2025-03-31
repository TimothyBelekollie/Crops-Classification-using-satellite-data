# Crop Classification Machine Learning Project


## Overview

This project is part of a crop classification challenge using machine learning models. It demonstrates advanced data preprocessing, feature engineering, and model optimization to achieve a strong F1 score of **0.8177**. The goal is to showcase expertise in handling real-world classification tasks and optimizing machine learning models.

## Prerequisites

Ensure you have the following dependencies installed:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Dataset

- **Train Dataset:** `train.csv`
- **Test Dataset:** `test.csv`

## Steps in the Project

### 1. Install Required Packages

```python
!pip install xgboost --quiet
```

### 2. Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
import ast
```

### 3. Load the Data

```python
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
```

### 4. Data Preprocessing & Feature Engineering

To improve model performance, the following steps were applied:

- Handling missing values
- Encoding categorical features
- Feature scaling using `StandardScaler`
- Feature selection based on importance

```python
def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return x
```

### 5. Model Training & Hyperparameter Tuning

Two models were trained and optimized:

- **Random Forest Classifier**
- **XGBoost Classifier**

Hyperparameter tuning was done using `GridSearchCV` to optimize performance:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
```

### 6. Model Evaluation

The best model achieved an **F1 score of 0.8177**, demonstrating its effectiveness in classification.

```python
y_pred = best_rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 7. Prediction on Test Data

```python
test_predictions = best_rf_model.predict(test_data)
```

## Key Achievements

✅ **Achieved an F1 score of 0.8177** using optimized hyperparameters. 
✅ Implemented advanced feature engineering to enhance predictive power. 
✅ Showcased expertise in machine learning model tuning and evaluation.

## Running the Project

1. Ensure dependencies are installed.
2. Run the notebook or script step by step.
3. Analyze model performance and predictions.

## Conclusion

This project implements a machine learning pipeline for crop classification. By leveraging feature engineering, hyperparameter tuning, and model evaluation techniques, it achieves a strong classification performance. This demonstrates practical expertise in deploying and optimizing ML models for real-world applications.

