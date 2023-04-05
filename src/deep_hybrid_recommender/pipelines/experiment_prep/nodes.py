"""
This is a boilerplate pipeline 'modeling_prep'
generated using Kedro 0.18.4
"""
from typing import List

import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split

import deep_hybrid_recommender.consts as cc


def split_the_data(data: pd.DataFrame, cat_encoders: dict, experiment_params: dict) -> List[th.Tensor]:
    """
    Splits the data into train/validation and test sets. The test set is used for final evaluation of the model.
    The train/validation set is used for cross-validation.

    Parameters
    ----------
    data: pd.DataFrame
        Data to split.
    cat_encoders: dict
        Dictionary of categorical encoders.
    experiment_params: dict
        Dictionary of experiment parameters, like test size.

    Returns
    -------
    List[th.Tensor]
        List of tensors with the train/validation and test sets.
    """
    columns_to_keep = [k for k in cat_encoders.keys()] + [cc.COL_USER_ID, cc.COL_ITEM_ID, cc.COL_OVERALL]
    data_subset = data[columns_to_keep]
    X, y = data_subset.drop(columns=cc.COL_OVERALL), data_subset[cc.COL_OVERALL]

    X_features, X_ids = X.drop(columns=[cc.COL_USER_ID, cc.COL_ITEM_ID]), X[[cc.COL_USER_ID, cc.COL_ITEM_ID]]

    X_train_f, X_test_f, X_train_ids, X_test_ids, y_train, y_test = train_test_split(X_features, X_ids, y, random_state=experiment_params[cc.PARAM_TRAIN_TEST_SEED],
                                                        test_size=experiment_params[cc.PARAM_TEST_SIZE])

    torch_data = [th.from_numpy(df.values) for df in [X_train_f, X_train_ids, X_test_f, X_test_ids, y_train, y_test] ]

    return torch_data
