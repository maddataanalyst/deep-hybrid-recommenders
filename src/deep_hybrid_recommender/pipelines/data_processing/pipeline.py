"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import node
from kedro.pipeline.modular_pipeline import Pipeline, pipeline
from .nodes import encode_categorical_data, prepare_categorical_encoders, reassign_user_item_ids


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_categorical_encoders,
            inputs=['amazon_data', 'params:categorical_col'],
            outputs='categorical_encoders',
            name='build_encoders'
        ),
        node(
            func=encode_categorical_data,
            inputs=['amazon_data', 'categorical_encoders', 'params:categorical_col'],
            outputs='encoded_data',
            name='encode_cat_data'
        ),
        node(
            func=reassign_user_item_ids,
            inputs=['encoded_data'],
            outputs=['uid_iid_reassigned_data', 'uid_iid_encoders'],
            name='reassign_uid_iid'
        )
    ],
    parameters={'params:categorical_col': 'categorical_col'},
    namespace='data_processing')
