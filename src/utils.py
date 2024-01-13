import pandas as pd
import json
import os
import mlflow

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

mlflow.autolog()

def load_params():
    """
    Load the configuration parameters 
    
    Returns:
    dict: dictionary with loaded parameters
    """
    
    params_path = os.path.join(os.getcwd(), 'conf/parameters.json')

    with open(params_path, 'r') as file:
        data_dict = json.load(file)

    return data_dict


def load_data(target_column):
    """
    Load California housing dataset from scikit-learn and organize it into a DataFrame.

    Parameters:
    - target_column (str): The name of the column to be used as the target variable in the DataFrame.

    Returns:
    - pandas.DataFrame: A DataFrame containing California housing dataset features and the specified target column.

    This function fetches the California housing dataset from scikit-learn, organizes it into a pandas DataFrame
    for better visualization, and includes the specified target column. The dataset includes various features related
    to California housing, and the target column represents the median house value for California districts.
    """

    california_housing = fetch_california_housing()

    # Create a DataFrame for better visualization (optional)
    data = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
    data[target_column] = california_housing.target

    return data


def preprocess_input(data, target_column, params_preprocessing):
    """
    Prepares the input data to train a machine learning model

    Parameters:
    data: pd.DataFrame
        The data to preprocess
    target_column: str
        The name of the target variable
    params_preprocessing: dict
        The parameters for the process

    Returns:
    pd.DataFrame
        the training features
    pd.DataFrame
        the testing features
    np.array:
        the training target
    np.array:
        the testing target
    """

    impute_method = params_preprocessing['impute_method']
    variable_selector = params_preprocessing['variable_selector']

    # Separate features from target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Impute null values
    num_imputer = SimpleImputer(strategy=impute_method)
    X_train_imputed = pd.DataFrame(num_imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(num_imputer.transform(X_test), columns=X_test.columns)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)

    # Feature Selection using RandomForestRegressor
    feature_selector = SelectFromModel(eval(variable_selector)())
    X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = feature_selector.transform(X_test_scaled)

    return X_train_selected, X_test_selected, y_train, y_test


def find_best_model(X_train, y_train, params_modeling):
    """
    Find the best hyperparameters for a given machine learning algorithm using grid search.

    Parameters:
    - X_train (numpy.ndarray or pandas.DataFrame): The feature matrix for training the model.
    - y_train (numpy.ndarray or pandas.Series): The target variable for training the model.
    - params_modeling (dict): Dictionary containing hyperparameter grid and algorithm information.
        - 'param_grid' (dict): Hyperparameter grid for grid search.
        - 'algorithm' (str): The machine learning algorithm to be used (e.g., 'RandomForestRegressor').

    Returns:
    - sklearn.model_selection.GridSearchCV: GridSearchCV object containing the best hyperparameters
      and the corresponding model trained on the input data.

    This function performs grid search with a specified machine learning algorithm (e.g., RandomForestRegressor)
    to find the best hyperparameters based on the negative mean squared error as the scoring metric. The hyperparameters
    and the corresponding mean squared error are printed, and the function returns the trained GridSearchCV object
    for further analysis or prediction.
    """

    # Hyperparameter grid for grid search
    param_grid=params_modeling['param_grid']
    algorithm=params_modeling['algorithm']

    # Grid Search with RandomForestRegressor
    rf_regressor = eval(algorithm)()

    with mlflow.start_run():
        grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and score
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best MSE:", grid_search.best_score_)

    return grid_search 