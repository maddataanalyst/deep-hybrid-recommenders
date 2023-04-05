import torch as th
import torch.nn as nn
import torch.utils.data as thu
import torchmetrics as tm
import pytorch_lightning as pl
import copy
import pandas as pd

from typing import Callable
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.model_selection import KFold
import deep_hybrid_recommender.consts as cc


class MetricsCallback(pl.Callback):

    def __init__(self):
        super().__init__()
        self.train_metrics = defaultdict(lambda: [])
        self.test_metrics = defaultdict(lambda: [])
        self.val_metrics = defaultdict(lambda: [])

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_metrics(trainer, self.train_metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_metrics(trainer, self.val_metrics)

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_metrics(trainer, self.test_metrics)

    def _log_metrics(self, trainer, collection):
        for k, v in trainer.callback_metrics.items():
            collection[k].append(v.item())


def cross_validate_model(
        model_builder_f: Callable[[], pl.LightningModule],
        X_train_ids: th.Tensor,
        X_train_features: th.Tensor,
        y_train: th.Tensor,
        k_folds: int,
        tensorboard_logger,
        model_name: str,
        batch_size: int = 64,
        epochs: int = 50,
        use_features: bool = True):

    train_metrics = defaultdict(lambda: [])
    val_metrics = defaultdict(lambda: [])

    kfold = KFold(k_folds)

    for train_idx, val_idx in tqdm(kfold.split(X_train_ids)):
        Xid_train_fold = X_train_ids[train_idx]
        uids_train = Xid_train_fold[:, 0]
        iids_train = Xid_train_fold[:, 1]
        X_feat_train_fold = X_train_features[train_idx]
        y_train_fold = y_train[train_idx].to(th.float)

        if use_features:
            train_ds = thu.TensorDataset(Xid_train_fold, X_feat_train_fold, y_train_fold)
        else:
            train_ds = thu.TensorDataset(uids_train, iids_train, y_train_fold)
        train_loader = thu.DataLoader(train_ds, batch_size=batch_size)

        Xid_val_fold = X_train_ids[val_idx]
        uids_val_fold = Xid_val_fold[:, 0]
        iids_val_fold = Xid_val_fold[:, 1]
        X_feat_val_fold = X_train_features[val_idx]
        y_val_fold = y_train[val_idx].to(th.float)

        if use_features:
            val_ds = thu.TensorDataset(Xid_val_fold, X_feat_val_fold, y_val_fold)
        else:
            val_ds = thu.TensorDataset(uids_val_fold, iids_val_fold, y_val_fold)
        val_loader = thu.DataLoader(val_ds, batch_size=batch_size)

        metrics_callback = MetricsCallback()
        trainer = pl.Trainer(logger=tensorboard_logger, max_epochs=epochs, enable_progress_bar=True,
                             enable_checkpointing=False, callbacks=[metrics_callback])

        model = model_builder_f()
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Store metrics from FINAL epoch (latest)
        for metric, values in metrics_callback.train_metrics.items():
            train_metrics[metric].append(values[-1])

        for metric, values in metrics_callback.val_metrics.items():
            val_metrics[metric].append(values[-1])

    train_metrics = pd.DataFrame(train_metrics)
    train_metrics[cc.MODEL_NAME] = model_name
    val_metrics = pd.DataFrame(val_metrics)
    val_metrics[cc.MODEL_NAME] = model_name

    # Metrics summary
    train_summary = train_metrics.agg(['mean', 'std']).to_dict()
    val_summary = val_metrics.agg(['mean', 'std']).to_dict()
    metrics_summary = train_summary | val_summary
    metrics_summary_flat = pd.json_normalize(metrics_summary, sep="_").to_dict(orient='records')[0]

    return train_metrics, val_metrics, metrics_summary_flat


def train_model_and_apply_on_test(
    model_builder_f: Callable[[], pl.LightningModule],
    X_train_ids: th.Tensor,
    X_train_features: th.Tensor,
    y_train: th.Tensor,
    X_test_ids: th.Tensor,
    X_test_features: th.Tensor,
    y_test: th.Tensor,
    tensorboard_logger,
    model_name: str,
    batch_size: int = 64,
    epochs: int = 50,
    use_features: bool = True):

    if use_features:
        train_ds = thu.TensorDataset(X_train_ids, X_train_features, y_train.to(th.float))
        test_ds = thu.TensorDataset(X_test_ids, X_test_features, y_test.to(th.float))
    else:
        uids_train = X_train_ids[:, 0]
        iids_train = X_train_ids[:, 1]
        train_ds = thu.TensorDataset(uids_train, iids_train, y_train.to(th.float))

        uids_test = X_test_ids[:, 0]
        iids_test = X_test_ids[:, 1]
        test_ds = thu.TensorDataset(uids_test, iids_test, y_test.to(th.float))

    train_loader = thu.DataLoader(train_ds, batch_size=batch_size)
    test_loader = thu.DataLoader(test_ds, batch_size=X_test_ids.size(0))

    model = model_builder_f()
    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(logger=tensorboard_logger, max_epochs=epochs, enable_progress_bar=True,
                         enable_checkpointing=False, callbacks=[metrics_callback])

    trainer.fit(model, train_dataloaders=train_loader)
    trainer.test(model, dataloaders=test_loader)

    test_metrics = {}
    for metric, values in metrics_callback.test_metrics.items():
        test_metrics[metric] = values[-1]

    return test_metrics


