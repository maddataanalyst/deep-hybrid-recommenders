import torch as th
import torch.nn as nn
import torchmetrics as tm
import torch_geometric as tg
import torch_geometric.data as tgd
import pytorch_lightning as pl
import pandas as pd

import deep_hybrid_recommender.consts as cc
import deep_hybrid_recommender.pipelines.experiment.models as models
import deep_hybrid_recommender.pipelines.experiment.experiment_commons as exc

from collections import defaultdict
from tqdm.auto import tqdm
from typing import List, Tuple, Callable
from sklearn.model_selection import KFold


def train_gnn_and_test(
        model: models.LitGNNRecommender,
        train_graph: tgd.HeteroData,
        test_graph: tgd.HeteroData,
        main_rel: Tuple[str, str, str],
        batch_size: int,
        max_epochs: int,
        neg_sampling_ratio: float,
        logdir: str,
        is_test: bool = True) -> dict:
    """
    Train a GNN model and test it on a test graph.
    Parameters
    ----------
    model: models.LitGNNRecommender
        A GNN model to be trained.
    train_graph: tgd.HeteroData
        A heterogeneous graph to be used for training.
    test_graph: tgd.HeteroData
        A heterogeneous graph to be used for testing.
    main_rel: Tuple[str, str, str]
        A tuple of (relation_name, source_node_type, target_node_type) that will be used as the main relation.
    batch_size: int
        Batch size.
    max_epochs: int
        Maximum number of epochs.
    neg_sampling_ratio: float
        Negative sampling ratio.
    logdir: str
        Directory to save logs.
    is_test: bool
        Is the method used on test set?

    Returns
    -------
    dict
        Test metrics.
    """
    train_loader = tg.loader.LinkNeighborLoader(
        data=train_graph,
        num_neighbors=[99, 99],
        edge_label_index=(main_rel, train_graph[main_rel].edge_label_index),
        edge_label=train_graph[main_rel].edge_label,
        neg_sampling_ratio=neg_sampling_ratio,
        shuffle=True,
        batch_size=batch_size,
    )
    eval_loader = tg.loader.LinkNeighborLoader(
        data=test_graph,
        num_neighbors=[99, 99],
        edge_label_index=(main_rel, test_graph[main_rel].edge_label_index),
        edge_label=test_graph[main_rel].edge_label,
        neg_sampling_ratio=0.0,
        shuffle=True,
        batch_size=batch_size,
    )
    tensorboard_logger = pl.loggers.tensorboard.TensorBoardLogger(save_dir=logdir)
    trainer = pl.Trainer(max_epochs=max_epochs, enable_progress_bar=True,
                         enable_checkpointing=False, accelerator='cpu', logger=tensorboard_logger)
    trainer.fit(model, train_dataloaders=train_loader)

    if is_test:
        results = trainer.test(model, dataloaders=eval_loader)
    else:
        results = trainer.validate(model, dataloaders=eval_loader)
    return results[0]


def cross_validate_graph_model(
    model_build_f: Callable[[], models.LitGNNRecommender],
    train_graph: tgd.HeteroData,
    main_rel: Tuple[str, str, str],
    batch_size: int,
    max_epochs: int,
    neg_sampling_ratio: float,
    logdir: str,
    kfold: int=10,
    model_name: str = "GNN"):
    """
    Graph-specific cross-validation function, that will split the graph into train and validation sets multiple times,
    keeping the main relation intact.

    Parameters
    ----------
    model_build_f: Callable[[], models.LitGNNRecommender]
        A function that builds a model, a factory function.
    train_graph: tgd.HeteroData
        A heterogeneous graph to be used for training.
    main_rel: Tuple[str, str, str]
        A tuple of (relation_name, source_node_type, target_node_type) that will be used as the main relation.
    batch_size: int
        Batch size.
    max_epochs: int
        Maximum number of epochs.
    neg_sampling_ratio: float
        Negative sampling ratio.
    logdir: str
        Directory to save logs.
    kfold: int
        Number of folds.
    model_name: str
        Model name.
    """

    splitter = KFold(n_splits=kfold)
    train_metrics = defaultdict(lambda: [])
    val_metrics = defaultdict(lambda: [])

    edge_label_idx = train_graph[main_rel].edge_label_index.t()
    edge_label = train_graph[main_rel].edge_label

    for batch_train, batch_val in splitter.split(edge_label_idx, edge_label):
        train_indices = edge_label_idx[batch_train]
        train_labels = edge_label[batch_train]

        val_indices = edge_label_idx[batch_val]
        val_labels = edge_label[batch_val]

        train_batch_graph = train_graph.clone()
        train_batch_graph[main_rel].edge_label_index = train_indices.t()
        train_batch_graph[main_rel].edge_label = train_labels

        val_batch_graph = train_graph.clone()
        val_batch_graph[main_rel].edge_label_index = val_indices.t()
        val_batch_graph[main_rel].edge_label = val_labels

        model = model_build_f()
        fold_val_metrics = train_gnn_and_test(
            model=model,
            train_graph=train_batch_graph,
            test_graph=val_batch_graph,
            main_rel=main_rel,
            batch_size=batch_size,
            max_epochs=max_epochs,
            neg_sampling_ratio=neg_sampling_ratio,
            logdir=logdir,
            is_test=False
        )
        for metric, val in fold_val_metrics.items():
            val_metrics[metric].append(val)

        with th.no_grad():
            fold_train_metrics = model._perform_prediction(train_batch_graph, "train", False)
            for metric, val in fold_train_metrics.items():
                train_metrics[metric].append(val)

    return exc.assemble_crossval_metrics(model_name, train_metrics, val_metrics)
