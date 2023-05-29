import enum
from typing import Tuple, Callable

import pytorch_lightning as pl
import torch as th
import torch.nn as nn
import torchmetrics as tm
import torch_geometric.nn as tgnn
import torch_geometric.data as tgd

from enum import Enum

def _build_subnet(
        nelems: int,
        nembedding: int,
        hidden_sizes: Tuple[int, ...],
        dropout: float = 0.0,
        activation = nn.ReLU) -> nn.Sequential:
    """
    Build a subnet for the collaborative filtering model.

    Parameters
    ----------
    nelems: int
        Number of elements in the input.
    nembedding: int
        Size of the embedding layer.
    hidden_sizes: Tuple[int, ...]
        Sizes of the hidden layers.
    dropout: float
        Dropout rate.
    activation: nn.Module
        Activation function.

    Returns
    -------
    nn.Sequential
        The subnet for specific task.
    """
    layers = []
    last_out = nelems
    if nembedding > 0:
        embedding = nn.Embedding(num_embeddings=nelems + 1, embedding_dim=nembedding)
        layers.append(embedding)
        last_out = nembedding
    for hidden_sz in hidden_sizes:
        layer = nn.Linear(in_features=last_out, out_features=hidden_sz)
        layers.append(layer)
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        last_out = hidden_sz
    return nn.Sequential(*layers)


class ColabFiltering(nn.Module):

    """Classic collaborative filtering model based on dot product of user and item embeddings (matrix factorization)."""

    def __init__(self, nusers: int, nitems: int, embed_sz: int = 25, hidden_sizes: Tuple[int, ...] = ()):
        super().__init__()
        self.nusers = nusers
        self.nitems = nitems
        self.hidden_sizes = hidden_sizes

        self.user_subnet = _build_subnet(self.nusers, embed_sz, hidden_sizes)
        self.item_subnet = _build_subnet(self.nitems, embed_sz, hidden_sizes)

    def forward(self, uid, iid):
        uid_embed = self.user_subnet(uid)
        iid_embed = self.item_subnet(iid)
        dotprod = th.sum(uid_embed * iid_embed, dim=-1)
        return th.relu(dotprod)


class LitColabFiltering(pl.LightningModule):

    """Lightning module for collaborative filtering model."""

    TRACKED_METRICS = ['MSE', 'MAE', 'MAPE']

    def __init__(self, n_users: int, n_items: int, embed_sz: int, hidden_sizes: Tuple[int, ...], learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = ColabFiltering(n_users, n_items, embed_sz, hidden_sizes)

    def _perform_prediction(self, batch, phase: str):
        uid, iid, y = batch
        yhat = self.model(uid, iid)
        mse = nn.functional.mse_loss(yhat, y)
        mape = tm.functional.mean_absolute_percentage_error(yhat, y)
        mae = tm.functional.mean_absolute_error(yhat, y)
        self.log(f"{phase}_MSE", mse, on_epoch=True, on_step=False)
        self.log(f"{phase}_MAPE", mape, on_epoch=True, on_step=False)
        self.log(f"{phase}_MAE", mae, on_epoch=True, on_step=False)
        return {"loss": mse, "mape": mape, "mae": mae}

    def training_step(self, batch, batch_index):
        self.model.train()
        return self._perform_prediction(batch, "train")

    def validation_step(self, batch, batch_index):
        self.model.eval()
        return self._perform_prediction(batch, "val")

    def test_step(self, batch, batch_index):
        self.model.eval()
        return self._perform_prediction(batch, "test")

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=self.learning_rate)


class DeepHybridRecommender(nn.Module):

    """Deep hybrid collaborative filtering model with user and item embeddings and categorical data embeddings."""

    def __init__(
            self,
            user_embeds: Tuple[int, int, Tuple[int, ...]],
            item_embeds: Tuple[int, int, Tuple[int, ...]],
            cat_col_embeds: Tuple[Tuple[int, int], ...],
            embed_concat_subnet_specs: Tuple[int, ...],
            dense_layers: Tuple[int, ...],
            inner_activation_fn = nn.Tanh,
            dropout: float=0.2):
        super().__init__()
        self.dropout = dropout
        self.inner_activation_fn = inner_activation_fn

        self.usubnet = _build_subnet(user_embeds[0], user_embeds[1], user_embeds[2], self.dropout, self.inner_activation_fn)
        self.isubnet = _build_subnet(item_embeds[0], item_embeds[1], item_embeds[2], self.dropout, self.inner_activation_fn)
        self.cat_data_subntes = nn.ModuleList()

        n_elems_total = user_embeds[2][-1] + item_embeds[2][-1]

        cat_elems_total = 0
        for (cat_nitems, cat_embed_sz) in cat_col_embeds:
            subnet = _build_subnet(cat_nitems, cat_embed_sz, [], self.dropout, self.inner_activation_fn)
            self.cat_data_subntes.append(subnet)
            subnet_last_out = cat_embed_sz
            cat_elems_total += subnet_last_out
        self.embed_concat_subnet = _build_subnet(cat_elems_total, 0, embed_concat_subnet_specs, self.dropout, self.inner_activation_fn)

        self.dense_cat = nn.Linear(in_features=int(cat_elems_total), out_features=8)
        n_elems_total += 8

        dense_modules = nn.ModuleList()
        last_out = n_elems_total
        for h_sz in dense_layers:
            dense_modules.append(nn.Linear(in_features=last_out, out_features=h_sz))
            dense_modules.append(self.inner_activation_fn())
            last_out = h_sz
        dense_modules.append(nn.Linear(in_features=last_out, out_features=1))
        dense_modules.append(nn.ReLU())
        self.dense_subnet = nn.Sequential(*dense_modules)

    def forward(self, uid: th.Tensor, iid: th.Tensor, X_cat: th.Tensor):
        u_output = self.usubnet(uid)
        i_output = self.isubnet(iid)
        cat_outputs = [self.cat_data_subntes[i](X_cat[:, i]) for i in range(X_cat.size(1))]
        cat_dense_out = self.dense_cat(th.cat(cat_outputs, dim=-1))

        all_data = th.cat([u_output, i_output, cat_dense_out], dim=-1)
        out = self.dense_subnet(all_data)
        return out


class LitDeepHybridRecommender(pl.LightningModule):

    """Lightning module for DeepHybridRecommender model."""

    TRACKED_METRICS = ['MSE', 'MAE', 'MAPE']

    def __init__(self,  user_embeds: Tuple[int, int, Tuple[int, ...]],
            item_embeds: Tuple[int, int, Tuple[int, ...]],
            cat_col_embeds: Tuple[Tuple[int, int], ...],
            embed_concat_subnet_specs: Tuple[int, ...],
            dense_layers: Tuple[int, ...],
            inner_activation_fn: nn.Module=nn.Tanh,
            dropout: float=0.2, learning_rate: float=0.01):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = DeepHybridRecommender(user_embeds, item_embeds, cat_col_embeds, embed_concat_subnet_specs, dense_layers, inner_activation_fn, dropout)

    def _perform_prediction(self, batch, phase: str):
        ids, features, y = batch
        yhat = self.model(ids[:, 0], ids[:, 1], features).squeeze()
        mse = nn.functional.mse_loss(yhat, y)
        mape = tm.functional.mean_absolute_percentage_error(yhat, y)
        mae = tm.functional.mean_absolute_error(yhat, y)
        self.log(f"{phase}_MSE", mse, on_epoch=True, on_step=False)
        self.log(f"{phase}_MAPE", mape, on_epoch=True, on_step=False)
        self.log(f"{phase}_MAE", mae, on_epoch=True, on_step=False)
        return {"loss": mse, "mape": mape, "mae": mae}

    def training_step(self, batch, batch_index):
        self.model.train()
        return self._perform_prediction(batch, "train")

    def validation_step(self, batch, batch_index):
        self.model.eval()
        return self._perform_prediction(batch, "val")

    def test_step(self, batch, batch_index):
        self.model.eval()
        return self._perform_prediction(batch, "test")

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=self.learning_rate)



class GNNConvType(enum.Enum):

    SAGE = 'Sage'
    GAT = 'GAT'
    GCN = 'GCN'
    EDGE = 'Edge'
    GIN = "GIN"


class LitGNNRecommender(pl.LightningModule):

    """
    Lightning module for GNNRecommender model, that utilizes a GNN to learn user and item embeddings and
    perform link predictions.
    """

    def __init__(
            self,
            nusers: int,
            uembed_dim: int,
            nitems: int,
            iembed_dim: int,
            graph:tgd.HeteroData,
            cat_col_embeds: Tuple[Tuple[int, int], ...],
            main_rel_name: Tuple[str, str, str],
            h_dim = 16,
            num_layers: int = 2,
            gnn_act_f: str = 'relu',
            subnet_act_f: nn.Module = nn.ReLU(),
            dropout: float = 0.2,
            use_item_features: bool = True,
            learning_rate: float = 0.001,
            gnn_type: GNNConvType = GNNConvType.SAGE,
            jk: str = None,
            crit = nn.MSELoss(),
            batch_size: int = 64):
        """
        Initializes the model.

        Parameters
        ----------
        nusers: int
            Number of users in the dataset.
        uembed_dim: int
            Dimensionality of the user embeddings.
        nitems: int
            Number of items in the dataset.
        iembed_dim: int
            Dimensionality of the item embeddings.
        graph: tgd.HeteroData
            Graph data structure containing the user-item interactions and the item features. This is so-called
            heterogeneous graph, where the nodes are of different types and the edges are of different types.
        cat_col_embeds: Tuple[Tuple[int, int], ...]
            Tuple of tuples, where each tuple contains the number of unique values and the embedding dimensionality for a
            categorical feature.
        main_rel_name: Tuple[str, str, str]
            Name of the main relation in the graph, which is the relation that contains the user-item interactions.
        h_dim: int
            Dimensionality of the hidden layers in the GNN.
        num_layers: int
            Number of layers in the GNN.
        gnn_act_f: str
            Name of the activation function used in the GNN.
        subnet_act_f: nn.Module
            Activation function used in the subnetworks.
        dropout: float
            Dropout rate.
        use_item_features: bool
            Whether to use item features or not.
        learning_rate: float
            Learning rate.
        gnn_type: GNNConvType
            Type of the GNN convolution.
        jk: str
            Type of the jump knowledge aggregation. If None, no jump knowledge aggregation is performed. Jumping knowledge
            aggregation is a technique that allows to aggregate information from all the layers of the GNN.
        crit: nn.Module
            Loss function.
        batch_size: int
            Batch size. It is needed for logging purposes. At this time, TorchGeometric used with PyTorchLightning
            is not able to log the loss function value without a batch size, so we need to do it manually.
        """
        super().__init__()
        self.main_rel_name = main_rel_name
        self.batch_size = batch_size
        self.crit = crit
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.uembed = nn.Embedding(num_embeddings=nusers, embedding_dim=uembed_dim)
        self.iembed = nn.Embedding(num_embeddings=nitems, embedding_dim=iembed_dim)
        self.use_item_features = use_item_features
        gnn = self._initialize_gnn(gnn_act_f, gnn_type, h_dim, jk, num_layers)
        self.gnn = tgnn.to_hetero(gnn, graph.metadata())

        if self.use_item_features:
            self.cat_data_subntes = nn.ModuleList()
            for (cat_nitems, cat_embed_sz) in cat_col_embeds:
                subnet = _build_subnet(cat_nitems, cat_embed_sz, [], self.dropout, subnet_act_f)
                self.cat_data_subntes.append(subnet)

    def _initialize_gnn(self, gnn_act_f, gnn_type, h_dim, jk, num_layers):
        if gnn_type == GNNConvType.SAGE:
            gnn = tgnn.GraphSAGE((-1, -1), h_dim, num_layers=num_layers, act=gnn_act_f, jk=jk)
        elif gnn_type == GNNConvType.GAT:
            gnn = tgnn.GAT((-1, -1), h_dim, num_layers=num_layers, act=gnn_act_f, v2=True, add_self_loops=False, jk=jk)
        elif gnn_type == GNNConvType.GIN:
            gnn = tgnn.GIN(-1, h_dim, num_layers=num_layers, act=gnn_act_f)
        elif gnn_type == GNNConvType.EDGE:
            gnn = tgnn.EdgeCNN((-1, -1), h_dim, num_layers=num_layers, act=gnn_act_f, jk=jk)
        elif gnn_type == GNNConvType.GCN:
            gnn = tgnn.GCN((-1, -1), h_dim, num_layers=num_layers, act=gnn_act_f, jk=jk)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        return gnn

    def forward(self, data: tgd.HeteroData) -> Tuple[th.Tensor, th.Tensor]:
        """
        Performs a forward pass through the model.

        Parameters
        ----------
        data: tgd.HeteroData
            Graph data structure containing the user-item interactions and the item features. This is so-called
            heterogeneous graph, where the nodes are of different types and the edges are of different types.

        Returns
        -------
        Tuple[th.Tensor, th.Tensor]
            Tuple of tensors containing the user-item embeddings and their dotproduct (actual prediction).
        """
        u_embed = self.uembed(data['user'].x)
        i_embed = self.iembed(data['item'].x)

        if self.use_item_features:
            features_mat = data['item'].feat
            fembeds = th.cat([self.cat_data_subntes[i](features_mat[:, i]) for i in range(features_mat.size(1))], dim=-1)
            item_data = th.cat([fembeds, i_embed.squeeze(1)], dim=-1)
        else:
            item_data = i_embed.squeeze(1)

        x_dict = {
            'user': u_embed.squeeze(1),
            'item': item_data,
        }
        out = self.gnn.forward(x_dict, data.edge_index_dict)
        superv_user = out['user'][data[self.main_rel_name].edge_label_index[0, :]]
        superv_movie = out['item'][data[self.main_rel_name].edge_label_index[1, :]]
        dotprod = (superv_user * superv_movie).sum(dim=-1)

        return out, dotprod


    def _perform_prediction(self, batch, phase: str, log: bool = True):
        _, dotprod = self.forward(batch)
        ytrain = batch[self.main_rel_name].edge_label.to(th.float)
        yhat_train = dotprod

        mse_loss = nn.functional.mse_loss(yhat_train, ytrain)
        mape_loss = tm.functional.mean_absolute_percentage_error(yhat_train, ytrain)
        mae_loss = tm.functional.mean_absolute_error(yhat_train, ytrain)
        if log:
            self.log(f"{phase}_MSE", mse_loss, batch_size=64)
            self.log(f"{phase}_MAPE", mape_loss, batch_size=64)
            self.log(f"{phase}_MAE", mae_loss, batch_size=64)

        return {"loss": mse_loss, "MAPE": mape_loss, "MAE": mae_loss}

    def training_step(self, batch, batch_index):
        self.train()
        return self._perform_prediction(batch, "train")

    def validation_step(self, batch, batch_index):
        self.eval()
        return self._perform_prediction(batch, "val")

    def test_step(self, batch, batch_index):
        self.eval()
        return self._perform_prediction(batch, "test")

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=self.learning_rate)
