"""
regression_cv.py

Regression models
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import  LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from imblearn.over_sampling import SMOTE 
import pandas as pd
import numpy as np
import itertools
import sys


def nearest_neighbor(train, val, k=5):
    """
    Run importance sampling nearest neighbors alg

    :param train: pd.DataFrame, train data, normalized
    :param val: pd.DataFrame, validation data, normalized to train
    :param k: <int>, number of neighbors, defaults to 5

    :return: np.array, the neighbors
    """
    
    # Find neighbors
    nn = NearestNeighbors(n_neighbors=k).fit(train)
    neighbors = np.unique(nn.kneighbors(X=val)[1].flatten())
    
    return neighbors


def scale_data(train, val, features, target, neighbors=None, smote=False):
    """"
    Scale data, run nearest neighbors/smote as needed

    :param train: pd.DataFrame, training data. Assume column called "data" specifies dataset,
                                "study_id" specifies ID
    :param val: pd.DataFrame, validation data. Assume "study_id" specifies study ID
    :param features: list<str>, the feature set
    :param target: <str>, the target col
    :param neighbors: <int>, whether to use neighbors, will be None otherwise
    :param smote: <bool>, whether to run smote or not

    :return: normalized train/validation data
    """
    # Drop all NA values
    train = train[features + [target, 'study_id', 'data']].dropna()

    # Setup standard scaler
    sc = StandardScaler()
    sc.fit(train[features])

    # Normalize
    train_norm = pd.DataFrame(sc.transform(train[features]), columns=features)
    train_norm[target] = train[target].values
    # Run smote if needed
    if smote:
        train_norm, train_target = SMOTE(random_state=42).fit_resample(
            train_norm[features], train_norm[target])
        train_norm[target] = train_target.values

    if val is not None:
        val = val[features + [target, 'study_id', 'data']].dropna()
        val_norm = pd.DataFrame(sc.transform(val[features]), columns=features)
        val_norm[target] = val[target].values
        val_norm['data'] = val['data'].values
        val_norm['study_id'] = val['study_id'].values

        # Weight if necessary
        if neighbors is not None:
            train_norm = train_norm.iloc[
                nearest_neighbor(train_norm[features], val_norm[features], k=neighbors), :
            ].reset_index(drop=True)
    else:
        val_norm = None

    return train_norm, val_norm


def train_validate_model(args):
    """"
    Train an sklearn model and test on validation set

    :param m: <str>, the sklearn model type
    :param params: <dict>, the parameters to use for the model
    :param train: pd.DataFrame, training data
    :param features: list<str>, the list of features to use
    :param target: <str>, the name of the column to predict
    :param val: pd.DataFrame, the validation data, leave blank if just training
    :param smote: <bool>, whether to use smote
    :param neighbors: <int>/None, whether to use neighbor matching

    :return: metric (model, res, true, pred) model, performance if val exists, predicted values
    """
    try:
        m, params, train, features, target, val, smote, neighbors, curr = args

        # Get data param
        if len(train.data.unique()) > 1:
            data = 'both'
        else:
            data = train.data.unique()[0]

        # Get model
        model = get_model(m, params)

        # Normalize data
        train, val = scale_data(
            train=train, val=val, features=features, target=target, neighbors=neighbors, smote=smote
        )

        # Train model
        model.fit(X=train[features], y=train[target])

        # If val exists validate
        if val is not None:
            y_pred = model.predict(val[features])
            mae = mean_absolute_error(y_true=val[target], y_pred=y_pred)
            print(
                curr, val.study_id.unique()[0], data, smote, neighbors, 
                target, m, params, np.round(mae, 3)
            )
            return model, mae, val[target].values, y_pred, val.study_id.values
        else:
            return model

    except Exception as e:
        print(e)
        sys.exit(1)


def get_loso_cv_data(data, features, target):
    """
    Get data to use for cross validation
    Will create folds by days

    :param data: pd.DataFrame, the data, assume has features, target, time_col, 'data', 'study_id' columns
    :param features: list<str>, the features
    :param target: <str>, the target col

    :return: list<pd.DataFrame>, list<pd.DataFrame>, the data to use for the cv (train, val)
    """
    # Drop NA
    data = data[features + [target, 'study_id', 'data', 'day']].dropna()

    # Create CV
    train = []
    val = []
    for s in data.study_id.unique():
        # Get all data not in chunk and add to train
        if data.loc[data['study_id'] == s, :].shape[0] >= 30:
            train.append(data.loc[data['study_id'] != s, :])
            val.append(data.loc[data['study_id'] == s, :])

    return train, val


def get_model(model_type, params):
    """
    Get a model with a specific parameter set

    :param model_type: <str>, the type of model
    :param params: <dict>, the model params

    :return: sklearn model, the model with those parameter settings
    """
    # Fill here
    if model_type == 'lr':
        model = LinearRegression(**params)
    elif model_type == 'ridge':
        model = Ridge(**params)
    elif model_type == 'lasso':
        model = Lasso(**params)
    elif model_type == 'elasticnet':
        model = ElasticNet(**params)
    elif model_type == 'gbt':
        model = GradientBoostingRegressor(**params)
    elif model_type == 'rf':
        model = RandomForestRegressor(**params)
    elif model_type == 'sv':
        model = SVR(**params)
    elif model_type == 'decisiontree':
        model = DecisionTreeRegressor(**params)
    elif model_type == 'knn':
        model = KNeighborsRegressor(**params)

    return model


def get_param_combinations(param_dict):
    """
    Get parameter combinations from a dictionary

    :param param_dict: dict<s:list>, the dictionary where parameters are each specified in lists
    :return list<dict>, a list of all the parameter combination dicts
    """
    # Get combinations
    param_combinations = list(itertools.product(*param_dict.values()))
    dict_keys = list(param_dict.keys())

    # Iterate through combinations
    return [dict(zip(dict_keys, v)) for v in param_combinations]
