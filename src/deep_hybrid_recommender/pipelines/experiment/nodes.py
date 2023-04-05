"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.18.5
"""
from typing import Tuple

import pandas as pd
import pingouin as pg
import plotly.graph_objs as go
import pytorch_lightning as pl
import torch as th
import torch.nn as nn

import deep_hybrid_recommender.consts as cc
from deep_hybrid_recommender.pipelines.experiment.experiment_commons import cross_validate_model, \
    train_model_and_apply_on_test
from deep_hybrid_recommender.pipelines.experiment.models import LitColabFiltering, LitDeepHybridRecommender

ACTIVATION2TORCH = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'leaky_relu': nn.LeakyReLU,
}


def crossval_collaboartive_filtering(
        X_train_ids: th.Tensor,
        y_train: th.Tensor,
        id_encoders: dict,
        model_params: dict
):
    """
    This function trains a collaborative filtering model and returns the train and validation metrics for each fold.
    Delegates execution to a dedicated, generic function.
    Parameters
    ----------
    X_train_ids
        A tensor of size (n_samples, 2) containing the user and item ids.
    y_train
        A tensor of size (n_samples, 1) containing the ratings.
    id_encoders
        A dictionary of id encoders.
    model_params
        A dictionary of model parameters.
    Returns
    -------
        Train and validation metrics for each fold.
    """
    nusers = id_encoders[cc.USER_ID_ENCODER].classes_.shape[0]
    nitems = id_encoders[cc.ITEM_ID_ENCODER].classes_.shape[0]

    embed_sz = model_params[cc.PARAM_ID_EMBED_SIZE]
    batch_size = model_params[cc.PARAM_BATCH_SIZE]
    lr = model_params[cc.PARAM_LEARNING_RATE]
    hidden_sizes = tuple(model_params[cc.PARAM_HIDDEN_SIZES])
    logdir = model_params[cc.PARAM_LOGDIR]
    max_epochs = model_params[cc.PARAM_MAX_EPOCHS]
    kfolds = model_params[cc.PARAM_KFOLD]

    model_builder_f = lambda: LitColabFiltering(nusers, nitems, embed_sz, hidden_sizes=hidden_sizes, learning_rate=lr)
    tensorboard_logger = pl.loggers.tensorboard.TensorBoardLogger(save_dir=logdir)

    return cross_validate_model(model_builder_f, X_train_ids, X_train_ids, y_train, kfolds,
                                tensorboard_logger, model_params[cc.MODEL_NAME], batch_size, max_epochs, False)


def crossval_deep_hybrid_recommender(
        X_train_ids: th.Tensor,
        X_train_features: th.Tensor,
        y_train: th.Tensor,
        id_encoders: dict,
        cat_encoders: dict,
        model_params: dict
):
    """
    This function trains a deep hybrid recommender model and returns the train and validation metrics for each fold.
    Delegates execution to a dedicated, generic function.

    Parameters
    ----------
    X_train_ids
        A tensor of size (n_samples, 2) containing the user and item ids.
    X_train_features
        A tensor of size (n_samples, n_features) containing the categorical features.
    y_train
        A tensor of size (n_samples, 1) containing the ratings.
    id_encoders
        A dictionary of id encoders.
    cat_encoders
        A dictionary of categorical encoders.
    model_params
        A dictionary of model parameters.
    Returns
    -------
        Train and validation metrics for each fold.
    """
    nusers = id_encoders[cc.USER_ID_ENCODER].classes_.shape[0]
    nitems = id_encoders[cc.ITEM_ID_ENCODER].classes_.shape[0]

    cat_encoders_classes = [len(enc.classes_) for _, enc in cat_encoders.items()]

    uid_subnet_specs = (nusers, *model_params[cc.PARAM_UID_SUBNET_SPECS])
    iid_subnet_specs = (nitems, *model_params[cc.PARAM_IID_SUBNET_SPECS])
    cat_subnet_specs = [(cat_encoders_classes[i], model_params[cc.PARAM_CAT_EMBEDDING_SPECS][i]) for i in
                        range(len(cat_encoders_classes))]
    embed_concat_subnet_specs = model_params[cc.PARAM_EMBED_CONCAT_DENSE_SPECS]
    dense_subnet_specs = model_params[cc.PARAM_DENSE_LAYERS]

    dropout = model_params[cc.PARAM_DROPOUT]

    batch_size = model_params[cc.PARAM_BATCH_SIZE]
    lr = model_params[cc.PARAM_LEARNING_RATE]
    logdir = model_params[cc.PARAM_LOGDIR]
    max_epochs = model_params[cc.PARAM_MAX_EPOCHS]
    activation_f = ACTIVATION2TORCH[model_params[cc.PARAM_INNER_ACTIVATION_F]]
    kfolds = model_params[cc.PARAM_KFOLD]

    model_builder_f = lambda: LitDeepHybridRecommender(uid_subnet_specs, iid_subnet_specs, cat_subnet_specs,
                                                       embed_concat_subnet_specs, dense_subnet_specs, activation_f,
                                                       learning_rate=lr, dropout=dropout)
    tensorboard_logger = pl.loggers.tensorboard.TensorBoardLogger(save_dir=logdir)

    return cross_validate_model(model_builder_f, X_train_ids, X_train_features, y_train, kfolds,
                                tensorboard_logger, model_params[cc.MODEL_NAME], batch_size, max_epochs, True)


def train_colab_filtering_and_test(
        X_train_ids: th.Tensor,
        y_train: th.Tensor,
        X_test_ids: th.Tensor,
        y_test: th.Tensor,
        id_encoders: dict,
        model_params: dict):
    """
    This function trains a collaborative filtering model and saves the predictions on the test set, as well as metrics.
    Delegates execution to a dedicated, generic function.

    Parameters
    ----------
    X_train_ids
        A tensor of size (n_samples, 2) containing the user and item ids.
    y_train
        A tensor of size (n_samples, 1) containing the ratings.
    X_test_ids
        A tensor of size (n_samples, 2) containing the user and item ids.
    y_test
        A tensor of size (n_samples, 1) containing the ratings.
    id_encoders
        A dictionary of id encoders.
    model_params
        A dictionary of model parameters.
    Returns
    -------
        Train and validation metrics for each fold.
    """
    nusers = id_encoders[cc.USER_ID_ENCODER].classes_.shape[0]
    nitems = id_encoders[cc.ITEM_ID_ENCODER].classes_.shape[0]

    embed_sz = model_params[cc.PARAM_ID_EMBED_SIZE]
    batch_size = model_params[cc.PARAM_BATCH_SIZE]
    lr = model_params[cc.PARAM_LEARNING_RATE]
    hidden_sizes = tuple(model_params[cc.PARAM_HIDDEN_SIZES])
    logdir = model_params[cc.PARAM_LOGDIR]
    max_epochs = model_params[cc.PARAM_MAX_EPOCHS]

    model_builder_f = lambda: LitColabFiltering(nusers, nitems, embed_sz, hidden_sizes=hidden_sizes, learning_rate=lr)
    tensorboard_logger = pl.loggers.tensorboard.TensorBoardLogger(save_dir=logdir)
    return train_model_and_apply_on_test(
        model_builder_f, X_train_ids, None, y_train, X_test_ids, None, y_test, tensorboard_logger,
        model_params[cc.MODEL_NAME], batch_size, max_epochs, False)


def train_deep_hybrid_recommender_and_test(
        X_train_ids: th.Tensor,
        X_train_features: th.Tensor,
        y_train: th.Tensor,
        X_test_ids: th.Tensor,
        X_test_features: th.Tensor,
        y_test: th.Tensor,
        id_encoders: dict,
        cat_encoders: dict,
        model_params: dict
):
    """"
    This function trains a deep hybrid recommender model and saves the predictions on the test set, as well as metrics.
    Delegates execution to a dedicated, generic function.

    Parameters
    ----------
    X_train_ids
        A tensor of size (n_samples, 2) containing the user and item ids.
    X_train_features
        A tensor of size (n_samples, n_features) containing the categorical features.
    y_train
        A tensor of size (n_samples, 1) containing the ratings.
    X_test_ids
        A tensor of size (n_samples, 2) containing the user and item ids.
    X_test_features
        A tensor of size (n_samples, n_features) containing the categorical features.
    y_test
        A tensor of size (n_samples, 1) containing the ratings.
    id_encoders
        A dictionary of id encoders.
    cat_encoders
        A dictionary of categorical encoders.
    model_params
        A dictionary of model parameters.
    Returns
    -------
        Train and validation metrics for each fold.
    """
    nusers = id_encoders[cc.USER_ID_ENCODER].classes_.shape[0]
    nitems = id_encoders[cc.ITEM_ID_ENCODER].classes_.shape[0]

    cat_encoders_classes = [len(enc.classes_) for _, enc in cat_encoders.items()]

    uid_subnet_specs = (nusers, *model_params[cc.PARAM_UID_SUBNET_SPECS])
    iid_subnet_specs = (nitems, *model_params[cc.PARAM_IID_SUBNET_SPECS])
    cat_subnet_specs = [(cat_encoders_classes[i], model_params[cc.PARAM_CAT_EMBEDDING_SPECS][i]) for i in
                        range(len(cat_encoders_classes))]
    embed_concat_subnet_specs = model_params[cc.PARAM_EMBED_CONCAT_DENSE_SPECS]
    dense_subnet_specs = model_params[cc.PARAM_DENSE_LAYERS]

    dropout = model_params[cc.PARAM_DROPOUT]

    batch_size = model_params[cc.PARAM_BATCH_SIZE]
    lr = model_params[cc.PARAM_LEARNING_RATE]
    logdir = model_params[cc.PARAM_LOGDIR]
    max_epochs = model_params[cc.PARAM_MAX_EPOCHS]
    activation_f = ACTIVATION2TORCH[model_params[cc.PARAM_INNER_ACTIVATION_F]]
    model_params[cc.PARAM_KFOLD]

    model_builder_f = lambda: LitDeepHybridRecommender(uid_subnet_specs, iid_subnet_specs, cat_subnet_specs,
                                                       embed_concat_subnet_specs, dense_subnet_specs, activation_f,
                                                       learning_rate=lr, dropout=dropout)
    tensorboard_logger = pl.loggers.tensorboard.TensorBoardLogger(save_dir=logdir)

    return train_model_and_apply_on_test(model_builder_f, X_train_ids, X_train_features, y_train, X_test_ids,
                                         X_test_features, y_test,
                                         tensorboard_logger, model_params[cc.MODEL_NAME], batch_size, max_epochs, True)


def analyze_results(
        colab_filtering_val_metrics: pd.DataFrame,
        deepl_colab_filtering_val_metrics: pd.DataFrame,
        deep_hybrid_rec_val_metrics: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure]:
    """
    This function performs a post-hoc analysis of the results obtained by the different models.
    It uses Kruskal Test to test whether the mean performance of the different models is significantly different.
    Kruskal test was selected because it is a non-parametric test that does not assume normality of the data.

    After the overall test, it performs a pairwise test to determine which models are significantly different with
    Bonferroni correction of p-values for multiple comparisons.

    The results are then plotted.

    Parameters
    ----------
    colab_filtering_val_metrics
        The validation metrics for the collaborative filtering model.
    deepl_colab_filtering_val_metrics
        The validation metrics for the deep collaborative filtering model.
    deep_hybrid_rec_val_metrics
        The validation metrics for the deep hybrid recommender model.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure]
        The results of the overall test, the results of the pairwise test, and the plots.
    """
    colab_filt_melted = colab_filtering_val_metrics.melt(id_vars='model_name', var_name='metric')
    deep_colab_filt_melted = deepl_colab_filtering_val_metrics.melt(id_vars='model_name', var_name='metric')
    deep_hybrid_melted = deep_hybrid_rec_val_metrics.melt(id_vars='model_name', var_name='metric')
    all_metrics = pd.concat([colab_filt_melted, deep_colab_filt_melted, deep_hybrid_melted], ignore_index=True)

    pairwise_comps = []
    overall_checks = []
    pairwise_figures = []
    overall_figures = []

    for metric in ['val_MAPE', 'val_MSE', 'val_MAE']:
        overall_compare, pairwise_compare = perform_statistical_comparison(all_metrics, metric)
        overall_compare['metric'] = metric
        pairwise_compare['metric'] = metric
        overall_checks.append(overall_compare)
        pairwise_compare = pairwise_compare.round(2)
        pairwise_comps.append(pairwise_compare)

        fig_pairwise = go.Figure(data=[go.Table(
            columnwidth=[90] * pairwise_compare.shape[1],
            header=dict(values=list(pairwise_compare.columns),
                        align='left'),
            cells=dict(values=[pairwise_compare[c] for c in pairwise_compare.columns],
                       font_size=9,
                       align='left'))
        ])
        pairwise_figures.append(fig_pairwise)

        fig_overall = go.Figure(data=[go.Table(
            columnwidth=[90] * overall_compare.shape[1],
            header=dict(values=list(overall_compare.columns),
                        align='left'),
            cells=dict(values=[overall_compare[c] for c in overall_compare.columns],
                       font_size=9,
                       align='left'))
        ])
        overall_figures.append(fig_overall)


    all_pairwise_comparisons = pd.concat(pairwise_comps, axis=0, ignore_index=True)
    all_overall_comparisons = pd.concat(overall_checks, axis=0, ignore_index=True)

    return all_metrics, all_pairwise_comparisons, all_overall_comparisons, pairwise_figures[0], pairwise_figures[1], \
    pairwise_figures[2], overall_figures[0], overall_figures[1], overall_figures[2]


def perform_statistical_comparison(all_metrics: pd.DataFrame, metric: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function performs a post-hoc analysis of the results obtained by the different models.
    It uses Kruskal Test to test whether the mean performance of the different models is significantly different.
    Kruskal test was selected because it is a non-parametric test that does not assume normality of the data.
    Later, the pairwise test is performed to determine which models are significantly different. Each pairwise
    comparison is corrected in terms of p-value using Bonferroni method..

    Parameters
    ----------
    all_metrics: pd.DataFrame
        A dataframe containing the validation metrics for all the models.
    metric: str
        The metric to be compared.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the results of the overall test and the pairwise test.
    """
    metric_sset = all_metrics.loc[all_metrics.metric == metric]
    overall_compare = pg.kruskal(metric_sset, dv='value', between='model_name')
    pairwise_compare = pg.pairwise_tests(metric_sset, dv='value', between='model_name', parametric=False,
                                         padjust='bonferroni')
    return overall_compare, pairwise_compare
