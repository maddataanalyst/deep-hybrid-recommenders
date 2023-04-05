"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    crossval_collaboartive_filtering,
    crossval_deep_hybrid_recommender,
    train_colab_filtering_and_test,
    train_deep_hybrid_recommender_and_test,
    analyze_results)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=crossval_collaboartive_filtering,
            inputs=[
                'experiment_prep.X_train_ids',
                'experiment_prep.y_train',
                'data_processing.uid_iid_encoders',
                'params:collaborative_filtering'],
            outputs=['colab_filtering_crossval_train_metrics', 'colab_filtering_crossval_val_metrics', 'colab_filt_crossval_summary'],
            name='crossval_colab_filtering'
        ),
        node(
            func=crossval_collaboartive_filtering,
            inputs=[
                'experiment_prep.X_train_ids',
                'experiment_prep.y_train',
                'data_processing.uid_iid_encoders',
                'params:deep_collaborative_filtering'],
            outputs=['deep_colab_filtering_crossval_train_metrics', 'deep_colab_filtering_crossval_val_metrics',
                     'deep_colab_filt_crossval_summary'],
            name='crossval_deep_colab_filtering'
        ),
        node(
            func=crossval_deep_hybrid_recommender,
            inputs=[
                'experiment_prep.X_train_ids',
                'experiment_prep.X_train_features',
                'experiment_prep.y_train',
                'data_processing.uid_iid_encoders',
                'data_processing.categorical_encoders',
                'params:deep_hybrid_recommender'],
            outputs=[
                'deep_hybrid_rec_crossval_train_metrics',
                'deep_hybrid_rec_crossval_val_metrics',
                'deep_hybrid_rec_crossval_summary'],
            name='crossval_deep_hybrid_rec'
        ),
        node(
            func=analyze_results,
            inputs=[
                'colab_filtering_crossval_val_metrics',
                'deep_colab_filtering_crossval_val_metrics',
                'deep_hybrid_rec_crossval_val_metrics'],
            outputs=[
                'all_metrics',
                'pairwise_comparisons',
                'overall_comparisons',
                'val_mape_table_plot_comparison',
                'val_mse_table_plot_comparison',
                'val_mae_table_plot_comparison',
                'val_mape_table_plot_overall',
                'val_mse_table_plot_overall',
                'val_mae_table_plot_overall'],
            name='analyze_results'
        ),
        node(
            func=train_colab_filtering_and_test,
            inputs=[
                'experiment_prep.X_train_ids',
                'experiment_prep.y_train',
                'experiment_prep.X_test_ids',
                'experiment_prep.y_test',
                'data_processing.uid_iid_encoders',
                'params:collaborative_filtering'],
            outputs='collaborative_filtering_test_metrics',
            name='train_colab_filt_and_test'
        ),
        node(
            func=train_colab_filtering_and_test,
            inputs=[
                'experiment_prep.X_train_ids',
                'experiment_prep.y_train',
                'experiment_prep.X_test_ids',
                'experiment_prep.y_test',
                'data_processing.uid_iid_encoders',
                'params:deep_collaborative_filtering'],
            outputs='deep_collaborative_filtering_test_metrics',
            name='train_deep_colab_filt_and_test'
        ),
        node(
            func=train_deep_hybrid_recommender_and_test,
            inputs=[
                'experiment_prep.X_train_ids',
                'experiment_prep.X_train_features',
                'experiment_prep.y_train',
                'experiment_prep.X_test_ids',
                'experiment_prep.X_test_features',
                'experiment_prep.y_test',
                'data_processing.uid_iid_encoders',
                'data_processing.categorical_encoders',
                'params:deep_hybrid_recommender'],
            outputs='deep_hybrid_rec_test_metrics',
            name='train_hybrid_rec_and_test'
        )
    ],
        namespace='experiment',
        parameters={
            'params:collaborative_filtering': 'collaborative_filtering',
            'params:deep_collaborative_filtering': 'deep_collaborative_filtering',
            'params:deep_hybrid_recommender': 'deep_hybrid_recommender'
        },
        inputs={
            'experiment_prep.X_train_features',
            'experiment_prep.X_train_ids',
            'experiment_prep.X_test_features',
            'experiment_prep.X_test_ids',
            'experiment_prep.y_train',
            'experiment_prep.y_test',
            'data_processing.uid_iid_encoders',
            'data_processing.categorical_encoders'
        })
