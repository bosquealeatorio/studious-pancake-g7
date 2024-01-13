import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the California Housing dataset
california_housing = fetch_california_housing()
target_column='target'

# Create a DataFrame for better visualization (optional)
data = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
data[target_column] = california_housing.target

# Separate features from target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute null values
num_imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(num_imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(num_imputer.transform(X_test), columns=X_test.columns)

# Scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)

# Feature Selection using RandomForestRegressor
feature_selector = SelectFromModel(RandomForestRegressor())
X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = feature_selector.transform(X_test_scaled)

# Hyperparameter grid for grid search
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10],
    'min_samples_split': [5],
    'min_samples_leaf': [4]
}

# Grid Search with RandomForestRegressor
rf_regressor = RandomForestRegressor()
grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_selected, y_train)

# Print the best hyperparameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best MSE:", grid_search.best_score_)

# Evaluate the model on the test set
X_test_selected = feature_selector.transform(X_test_scaled)
y_pred = grid_search.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)

