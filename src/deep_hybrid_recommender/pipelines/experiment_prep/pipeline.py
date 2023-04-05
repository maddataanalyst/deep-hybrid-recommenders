"""
This is a boilerplate pipeline 'modeling_prep'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_the_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_the_data,
            inputs=[
                'data_processing.uid_iid_reassigned_data',
                'data_processing.categorical_encoders',
                'params:experiment_prep'],
            outputs=[
                'X_train_features',
                'X_train_ids',
                'X_test_features',
                'X_test_ids',
                'y_train',
                'y_test'],
            name='split_the_data'
        )
    ],
        namespace='experiment_prep',
        parameters={'params:experiment_prep': 'experiment_prep'},
        inputs={
            'data_processing.uid_iid_reassigned_data',
            'data_processing.categorical_encoders'
        }
    )
