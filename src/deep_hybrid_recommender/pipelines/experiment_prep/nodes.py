"""
This is a boilerplate pipeline 'modeling_prep'
generated using Kedro 0.18.4
"""

from typing import List

import pandas as pd
import torch as th
import torch_geometric as tg
import torch_geometric.nn as tgnn
import torch_geometric.data as tgd
import torchmetrics as tm
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
    data_subset = data[columns_to_keep].drop_duplicates()
    X, y = data_subset.drop(columns=cc.COL_OVERALL), data_subset[cc.COL_OVERALL]

    X_features, X_ids = X.drop(columns=[cc.COL_USER_ID, cc.COL_ITEM_ID]), X[[cc.COL_USER_ID, cc.COL_ITEM_ID]]

    # Split user-item pairs to train and test without overlapping users and items



    X_train_f, X_test_f, X_train_ids, X_test_ids, y_train, y_test = train_test_split(X_features, X_ids, y, random_state=experiment_params[cc.PARAM_TRAIN_TEST_SEED],
                                                        test_size=experiment_params[cc.PARAM_TEST_SIZE])

    X_train_stacked = pd.concat((X_train_f.reset_index(drop=True), X_train_ids.reset_index(drop=True)),axis=1)
    X_train_stacked[cc.COL_OVERALL] = y_train.values
    X_train_stacked_vs_test = X_train_stacked.merge(X_test_ids, on=[cc.COL_USER_ID, cc.COL_ITEM_ID], how='left', indicator='exists')
    X_train_only = X_train_stacked_vs_test[X_train_stacked_vs_test['exists'] == 'left_only'].drop_duplicates()

    X_train_f = X_train_only.drop(columns=[cc.COL_USER_ID, cc.COL_ITEM_ID, cc.COL_OVERALL, 'exists'])
    X_train_ids = X_train_only[[cc.COL_USER_ID, cc.COL_ITEM_ID]]
    y_train = X_train_only[cc.COL_OVERALL]

    torch_data = [th.from_numpy(df.values) for df in [X_train_f, X_train_ids, X_test_f, X_test_ids, y_train, y_test] ]

    return torch_data


def build_graph_data(
        X_train_ids: th.Tensor,
        X_train_f: th.Tensor,
        X_test_ids: th.Tensor,
        X_test_f: th.Tensor,
        y_train: th.Tensor,
        y_test: th.Tensor,
        id_encoders: dict,
):
    # Extract to common function
    train_df = _build_df_from_X_y_ids(X_train_f, X_train_ids, y_train)
    test_df = _build_df_from_X_y_ids(X_test_f, X_test_ids, y_test)

    uid = id_encoders['userid_encoder']
    iid = id_encoders['itemid_encoder']

    # Full graph df is needed for final training data and prediction on test - to memorize full structure
    # and process test nodes only in this context
    full_graph_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Build separate train/test tensors for edges and labels
    # Dim train edges: (2 x num_edges on train data)
    # Dim train labels: (num_edges on train data)
    train_edges, train_labels = _build_tensors(train_df)

    # Dim test edges: (2 x num_edges on test data)
    # Dim test labels: (num_edges on test data)
    test_edges, test_labels = _build_tensors(test_df)

    # Full graph edges
    # Dim full graph edges: (2 x num_edges on full graph)
    full_graph_edges = th.tensor(full_graph_df[['uid', 'iid']].values).to(th.long)

    # Unique item features per item.
    # Dim (num_items x num_features)
    item_features = full_graph_df[['iid'] + [c for c in full_graph_df.columns if 'feature' in c]]
    item_features = th.tensor(item_features.groupby('iid').first().values).to(th.long)

    # User "features" are just user ids
    all_user_ids_array = th.arange(0, len(uid.classes_)).long()

    # First item "features" are just item ids
    all_item_ids_array = th.arange(0, len(iid.classes_)).long()

    # Build train and test graphs
    hetero_graph_train = _initialize_graph(all_user_ids_array, all_item_ids_array, item_features)
    _add_edges_labels_and_label_idx(hetero_graph_train, train_edges, train_labels, train_edges)

    hetero_graph_test = _initialize_graph(all_user_ids_array, all_item_ids_array, item_features)
    _add_edges_labels_and_label_idx(hetero_graph_test, full_graph_edges, test_labels, test_edges)

    return hetero_graph_train, hetero_graph_test


def _initialize_graph(all_user_ids_array: th.Tensor, all_item_ids_array: th.Tensor, item_features: th.Tensor) -> tgd.HeteroData:
    """
    Builds a graph with user and item nodes and edges between them. Initializes graph structure WITHOUT labels for supervision.

    Parameters
    ----------
    all_user_ids_array: th.Tensor
        Consecutive user ids array. Dim: (num users x 1)
    all_item_ids_array: th.Tensor
        Consecutive item ids array. Dim (num items x 1)
    item_features: th.Tensor
        Item features matrix. Dim (num items x num features)

    Returns
    -------
    tgd.HeteroData
        Heterogenous graph
    """
    hetero_graph = tgd.HeteroData()
    hetero_graph['user'].x = all_user_ids_array.unsqueeze(1).to(th.long)
    hetero_graph['item'].x = all_item_ids_array.unsqueeze(1).to(th.long)
    hetero_graph['item'].feat = item_features

    return hetero_graph


def _add_edges_labels_and_label_idx(graph: tgd.HeteroData, struct_edges: th.Tensor, edge_labels: th.Tensor, edge_label_idx: th.Tensor):
    """
    Adds edges, labels and label indices to the graph - supervision elements for training.

    Parameters
    ----------
    graph: tgd.HeteroData
        Heterogenous graph data.
    struct_edges: th.Tensor
        Edges of the graph, including all its nodes. Dim (2 x num_edges)
    edge_labels: th.Tensor
        Supervision edges labels. Dim (num_superv_edges)
    edge_label_idx: th.Tensor
        Indices of supervision edges in the graph. Dim (2 x num_superv_edges)

    Returns
    -------

    """
    graph['user', 'rates', 'item'].edge_index = struct_edges.t().squeeze()
    graph['user', 'rates', 'item'].edge_label_index = edge_label_idx.t().squeeze()
    graph['user', 'rates', 'item'].edge_label = edge_labels.to(th.float)

    graph['item', 'rev_rates', 'user'].edge_index = struct_edges.t().flip(0).squeeze()


def _build_tensors(df):
    edges = th.tensor(df[['uid', 'iid']].values).to(th.long)
    labels = th.tensor(df['y'].values).to(th.float)
    return edges, labels


def _build_df_from_X_y_ids(X_f: th.Tensor, X_ids: th.Tensor, y: th.Tensor) -> pd.DataFrame:
    """
    Builds a dataframe from the tensors with features, ids and target, so that it can be used for graph structures
    building.

    Parameters
    ----------
    X_f: th.Tensor
        Tensor with features.
    X_ids: th.Tensor
        Tensor with ids.
    y: th.Tensor
        Tensor with target

    """
    df = pd.DataFrame(X_ids.numpy(), columns=['uid', 'iid'])
    for i in range(X_f.size(1)):
        fname = f'feature{i}'
        df[fname] = X_f[:, i].numpy()
    df['y'] = y.numpy()
    df = df.drop_duplicates()
    return df
