"""
Data preprocessing functions that will adjust the input for recommender systems.
"""
from typing import Tuple

import pandas as pd
import logging

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from tqdm.auto import tqdm

import deep_hybrid_recommender.consts as cc

def prepare_categorical_encoders(data: pd.DataFrame, col_name: str) -> dict:
    """
    Prepares categorical encoders for a column in a dataframe.
    In this dataset, the column is a dictionary with multiple keys and values, each indicating
    a category and a value for that category.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe with the column to be encoded
    col_name: str
        Name of the column to be encoded. This column should be a dictionary with multiple keys and values.

    Returns
    -------
    dict
        Dictionary with the encoders for each category in the column.

    """
    unique_values = defaultdict(lambda: set())
    for col in data[col_name]:
        if pd.isnull(col):
            continue
        else:
            for k, v in col.items():
                unique_values[k.replace(":", "")].add(v)

    logging.info(f"Found unique vals for column: {col_name}")
    for k, v in unique_values.items():
        logging.info(f"{k}: {len(v)}")

    col_encoders = {}
    for category, values in tqdm(unique_values.items()):
        vals = list(values)
        vals += [cc.MISSING_VAL]
        le = LabelEncoder()
        le.fit(vals)
        col_encoders[category] = le

    return col_encoders

def encode_categorical_data(data: pd.DataFrame, encoders: dict, col_name: str) -> pd.DataFrame:
    """
    Encodes categorical data in a column of a dataframe using previously defined categorical encoders.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe with the column to be encoded
    encoders: dict
        Dictionary with the encoders for each category in the column.
    col_name: str
        Name of the column to be encoded. This column should be a dictionary with multiple keys and values.

    Returns
    -------
    pd.DataFrame
        Dataframe with the encoded column.
    """
    encoded_vals = {k: [] for k in encoders.keys()}

    for col in tqdm(data[col_name]):
        if pd.isnull(col):
            for cat in encoders.keys():
                v = encoders[cat].transform([cc.MISSING_VAL])[0]
                encoded_vals[cat].append(v)
        else:
            for cat in encoders.keys():
                cat_with_colon = cat + ":"
                raw_val = cc.MISSING_VAL
                if cat_with_colon in col:
                    raw_val = col[cat_with_colon]
                elif cat in col:
                    raw_val = col[cat]
                le = encoders[cat]
                v = le.transform([raw_val])[0]
                encoded_vals[cat].append(v)

    for k, vals in encoded_vals.items():
        data[k] = vals
    return data


def reassign_user_item_ids(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Reassigns user and item ids to be integers starting from 0. It is needed for the embedding layers in the networks
    to properly allocate the lookup matrix/tables.

    Parameters
    ----------
    data: pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Dataframe with the new user and item ids where each id is an integer starting from 0 and is a set of
        consecutive integers.
    """
    uencoder = LabelEncoder()
    iencoder = LabelEncoder()

    uencoded = uencoder.fit_transform(data.reviewerID)
    iencoded = iencoder.fit_transform(data.asin)

    data[cc.COL_USER_ID] = uencoded
    data[cc.COL_ITEM_ID] = iencoded

    return data, {cc.USER_ID_ENCODER: uencoder, cc.ITEM_ID_ENCODER: iencoder}