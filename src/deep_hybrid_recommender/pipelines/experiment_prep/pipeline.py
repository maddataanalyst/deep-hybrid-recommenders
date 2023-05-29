"""
This is a boilerplate pipeline 'modeling_prep'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_the_data, build_graph_data


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
        ),
        node(
            func=build_graph_data,
            inputs=[
                'X_train_ids',
                'X_train_features',
                'X_test_ids',
                'X_test_features',
                'y_train',
                'y_test',
                'data_processing.uid_iid_encoders'
            ],
            outputs=['hetero_train_graph', 'hetero_test_graph'],
            name='build_graph_data'
        )
    ],
        namespace='experiment_prep',
        parameters={'params:experiment_prep': 'experiment_prep'},
        inputs={
            'data_processing.uid_iid_reassigned_data',
            'data_processing.categorical_encoders',
            'data_processing.uid_iid_encoders'
        }
    )
