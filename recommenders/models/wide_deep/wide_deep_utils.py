# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import math
import logging
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
)

logger = logging.getLogger(__name__)


class WideDeepModel(nn.Module):
    """Wide & Deep model for recommendation.

    The wide component is a linear model over sparse features (user, item, and
    their crossed interaction) that captures memorization.  The deep component
    is a DNN over dense embeddings that captures generalization.

    Reference: Cheng et al., "Wide & Deep Learning for Recommender Systems", 2016
    https://arxiv.org/abs/1606.07792

    Args:
        users (iterable): Distinct user IDs (vocabulary).
        items (iterable): Distinct item IDs (vocabulary).
        model_type (str): ``"wide"``, ``"deep"``, or ``"wide_deep"``.
        crossed_feat_dim (int): Hash bucket size for the crossed user×item
            feature (wide).
        user_dim (int): User embedding dimension (deep).
        item_dim (int): Item embedding dimension (deep).
        item_feat_shape (int or None): Dimension of item feature vector.
            ``None`` means no item features.
        dnn_hidden_units (tuple of int): Hidden layer sizes for the deep
            component.
        dnn_dropout (float): Dropout rate for the deep component.
        dnn_batch_norm (bool): Whether to use batch normalization in the deep
            component.
        user_col (str): User column name in DataFrames.
        item_col (str): Item column name in DataFrames.
        item_feat_col (str or None): Item feature column name in DataFrames.
        seed (int or None): Random seed for reproducibility.
    """

    def __init__(
        self,
        users,
        items,
        model_type="wide_deep",
        crossed_feat_dim=1000,
        user_dim=8,
        item_dim=8,
        item_feat_shape=None,
        dnn_hidden_units=(128, 128),
        dnn_dropout=0.0,
        dnn_batch_norm=True,
        user_col=DEFAULT_USER_COL,
        item_col=DEFAULT_ITEM_COL,
        item_feat_col=None,
        seed=None,
    ):
        super().__init__()

        if model_type not in ("wide", "deep", "wide_deep"):
            raise ValueError("model_type must be 'wide', 'deep', or 'wide_deep'")

        if seed is not None:
            torch.manual_seed(seed)

        # Vocabularies and column config
        self.users = list(users)
        self.items = list(items)
        self.user_col = user_col
        self.item_col = item_col
        self.item_feat_col = item_feat_col

        self.model_type = model_type
        self.n_users = len(self.users)
        self.n_items = len(self.items)
        self.crossed_feat_dim = crossed_feat_dim

        # Training state (populated by fit())
        self._train_seen_pairs = None
        self._item_features = None

        # --- Wide component ---
        if model_type in ("wide", "wide_deep"):
            self.wide_user = nn.Embedding(self.n_users, 1)
            self.wide_item = nn.Embedding(self.n_items, 1)
            self.wide_cross = nn.Embedding(crossed_feat_dim, 1)
            self.wide_bias = nn.Parameter(torch.zeros(1))
            nn.init.zeros_(self.wide_user.weight)
            nn.init.zeros_(self.wide_item.weight)
            nn.init.zeros_(self.wide_cross.weight)

        # --- Deep component ---
        if model_type in ("deep", "wide_deep"):
            self.deep_user = nn.Embedding(
                self.n_users, user_dim, max_norm=user_dim**0.5
            )
            self.deep_item = nn.Embedding(
                self.n_items, item_dim, max_norm=item_dim**0.5
            )
            # Truncated_normal initializer: stddev = 1/sqrt(dim)
            nn.init.trunc_normal_(
                self.deep_user.weight, std=1.0 / math.sqrt(user_dim)
            )
            nn.init.trunc_normal_(
                self.deep_item.weight, std=1.0 / math.sqrt(item_dim)
            )

            dnn_input_dim = user_dim + item_dim
            if item_feat_shape is not None:
                dnn_input_dim += item_feat_shape
            self.item_feat_shape = item_feat_shape

            # Linear -> ReLU -> BatchNorm -> Dropout
            layers = []
            in_dim = dnn_input_dim
            for units in dnn_hidden_units:
                layers.append(nn.Linear(in_dim, units))
                layers.append(nn.ReLU())
                if dnn_batch_norm:
                    layers.append(nn.BatchNorm1d(units))
                if dnn_dropout > 0:
                    layers.append(nn.Dropout(dnn_dropout))
                in_dim = units
            layers.append(nn.Linear(in_dim, 1))
            self.dnn = nn.Sequential(*layers)

    def _cross_hash(self, user_ids, item_ids, bucket_size):
        """Deterministic hash of (user, item) pairs into ``[0, bucket_size)``."""
        return (
            (user_ids.long() * self.n_items + item_ids.long()) % bucket_size
        ).abs()

    def forward(self, user_ids, item_ids, item_feats=None):
        """Forward pass.

        Args:
            user_ids (torch.LongTensor): User index tensor of shape ``(batch,)``.
            item_ids (torch.LongTensor): Item index tensor of shape ``(batch,)``.
            item_feats (torch.FloatTensor or None): Item feature tensor of shape
                ``(batch, item_feat_shape)``.  Required when the model has a deep
                component and ``item_feat_shape`` was set at construction time.

        Returns:
            torch.Tensor: Predicted ratings of shape ``(batch,)``.
        """
        out = torch.zeros(user_ids.size(0), device=user_ids.device)

        if self.model_type in ("wide", "wide_deep"):
            cross_idx = self._cross_hash(user_ids, item_ids, self.crossed_feat_dim)
            wide_out = (
                self.wide_user(user_ids).squeeze(-1)
                + self.wide_item(item_ids).squeeze(-1)
                + self.wide_cross(cross_idx).squeeze(-1)
                + self.wide_bias
            )
            out = out + wide_out

        if self.model_type in ("deep", "wide_deep"):
            u_emb = self.deep_user(user_ids)
            i_emb = self.deep_item(item_ids)
            if item_feats is not None:
                dnn_input = torch.cat([u_emb, i_emb, item_feats], dim=-1)
            else:
                dnn_input = torch.cat([u_emb, i_emb], dim=-1)
            deep_out = self.dnn(dnn_input).squeeze(-1)
            out = out + deep_out

        return out

    # ------------------------------------------------------------------
    # Public API: fit, predict, recommend_k_items
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df,
        n_epochs=10,
        batch_size=32,
        y_col=DEFAULT_RATING_COL,
        wide_optimizer="adagrad",
        wide_optimizer_lr=0.01,
        deep_optimizer="adadelta",
        deep_optimizer_lr=0.01,
        seed=None,
        eval_fn=None,
        eval_every_n_epochs=None,
        log_every_n_epochs=1,
    ):
        """Train the model.

        Args:
            train_df (pd.DataFrame): Training data.
            n_epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
            y_col (str): Label column name.
            wide_optimizer (str): Optimizer name for wide parameters
                (e.g. ``"adagrad"``).
            wide_optimizer_lr (float): Learning rate for the wide optimizer.
            deep_optimizer (str): Optimizer name for deep parameters
                (e.g. ``"adadelta"``).
            deep_optimizer_lr (float): Learning rate for the deep optimizer.
            seed (int or None): Random seed.
            eval_fn (callable or None): ``eval_fn(model, epoch)`` called every
                *eval_every_n_epochs* epochs.
            eval_every_n_epochs (int or None): Evaluation frequency in epochs.
            log_every_n_epochs (int): How often to log training loss.

        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Store training interactions for recommend_k_items(remove_seen=True)
        self._train_seen_pairs = set(
            zip(train_df[self.user_col], train_df[self.item_col])
        )
        if self.item_feat_col is not None and self.item_feat_col in train_df.columns:
            feat_df = train_df.drop_duplicates(self.item_col)
            self._item_features = dict(
                zip(feat_df[self.item_col], feat_df[self.item_feat_col])
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Build optimizers *after* .to(device) so parameters are on the right device
        wide_opt = None
        deep_opt = None
        if self.model_type in ("wide", "wide_deep"):
            wide_params = (
                list(self.wide_user.parameters())
                + list(self.wide_item.parameters())
                + list(self.wide_cross.parameters())
                + [self.wide_bias]
            )
            wide_opt = _build_optimizer(
                wide_optimizer, wide_params, lr=wide_optimizer_lr
            )
        if self.model_type in ("deep", "wide_deep"):
            deep_params = (
                list(self.deep_user.parameters())
                + list(self.deep_item.parameters())
                + list(self.dnn.parameters())
            )
            deep_opt = _build_optimizer(
                deep_optimizer, deep_params, lr=deep_optimizer_lr
            )

        # Dataset and DataLoader
        dataset = _WideDeepDataset(
            train_df,
            self.users,
            self.items,
            user_col=self.user_col,
            item_col=self.item_col,
            item_feat_col=self.item_feat_col,
            y_col=y_col,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        loss_fn = nn.MSELoss()
        self.train()

        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                uid, iid, feat, rating = [b.to(device) for b in batch]
                item_feats = feat if feat.numel() > 0 else None

                # Forward
                pred = self(uid, iid, item_feats)
                loss = loss_fn(pred, rating)

                # Backward
                if wide_opt is not None:
                    wide_opt.zero_grad()
                if deep_opt is not None:
                    deep_opt.zero_grad()

                loss.backward()

                if wide_opt is not None:
                    wide_opt.step()
                if deep_opt is not None:
                    deep_opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if epoch % log_every_n_epochs == 0:
                logger.info(
                    "Epoch %d/%d – loss = %.4f", epoch, n_epochs, avg_loss
                )

            if (
                eval_fn is not None
                and eval_every_n_epochs
                and epoch % eval_every_n_epochs == 0
            ):
                self.eval()
                eval_fn(self, epoch)
                self.train()

    def predict(self, df, batch_size=256):
        """Generate predictions for the given dataframe.

        Args:
            df (pd.DataFrame): Data to predict on.  Must contain
                ``user_col`` and ``item_col`` columns (and ``item_feat_col``
                if the model uses item features).
            batch_size (int): Batch size for prediction.

        Returns:
            list[float]: Predicted values.
        """
        device = next(self.parameters()).device
        self.eval()

        dataset = _WideDeepDataset(
            df,
            self.users,
            self.items,
            user_col=self.user_col,
            item_col=self.item_col,
            item_feat_col=self.item_feat_col,
            y_col=None,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in loader:
                uid, iid, feat = [b.to(device) for b in batch]
                item_feats = feat if feat.numel() > 0 else None
                out = self(uid, iid, item_feats)
                preds.extend(out.cpu().tolist())

        return preds

    def recommend_k_items(self, test, top_k=10, remove_seen=True):
        """Recommend top-k items for each user in the test set.

        Args:
            test (pd.DataFrame): Test data containing at least the user column.
            top_k (int): Number of items to recommend per user.
            remove_seen (bool): Whether to exclude items the user interacted
                with during training.

        Returns:
            pd.DataFrame: Top-k recommendations with columns
            ``[user_col, item_col, DEFAULT_PREDICTION_COL]``.
        """
        if remove_seen and self._train_seen_pairs is None:
            raise RuntimeError("Must call fit() before recommend_k_items()")

        test_users = test[self.user_col].unique()

        # Build all (user, item) candidate pairs
        pairs = list(product(test_users, self.items))
        candidate_df = pd.DataFrame(pairs, columns=[self.user_col, self.item_col])

        # Attach item features if applicable (only keep items that have features)
        if self.item_feat_col is not None and self._item_features is not None:
            items_with_feats = set(self._item_features.keys())
            candidate_df = candidate_df[
                candidate_df[self.item_col].isin(items_with_feats)
            ].reset_index(drop=True)
            candidate_df[self.item_feat_col] = candidate_df[self.item_col].map(
                self._item_features
            )

        # Remove items seen during training
        if remove_seen and self._train_seen_pairs is not None:
            seen_df = pd.DataFrame(
                list(self._train_seen_pairs),
                columns=[self.user_col, self.item_col],
            )
            candidate_df = candidate_df.merge(
                seen_df,
                on=[self.user_col, self.item_col],
                how="left",
                indicator=True,
            )
            candidate_df = (
                candidate_df[candidate_df["_merge"] == "left_only"]
                .drop("_merge", axis=1)
                .reset_index(drop=True)
            )

        # Predict scores
        scores = self.predict(candidate_df)
        candidate_df[DEFAULT_PREDICTION_COL] = scores

        # Select top-k per user
        top_k_df = (
            candidate_df.sort_values(DEFAULT_PREDICTION_COL, ascending=False)
            .groupby(self.user_col)
            .head(top_k)
            .reset_index(drop=True)
        )

        return top_k_df[[self.user_col, self.item_col, DEFAULT_PREDICTION_COL]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _WideDeepDataset(Dataset):
    """PyTorch Dataset for the Wide & Deep model.

    Maps raw user / item IDs to contiguous indices based on the provided
    vocabularies.

    Args:
        df (pd.DataFrame): Data with user, item, optional item-feature, and
            rating columns.
        users (list): Ordered user ID vocabulary (index = embedding row).
        items (list): Ordered item ID vocabulary.
        user_col (str): User column name.
        item_col (str): Item column name.
        item_feat_col (str or None): Item feature column name.
        y_col (str or None): Rating / label column name.
    """

    def __init__(
        self,
        df,
        users,
        items,
        user_col=DEFAULT_USER_COL,
        item_col=DEFAULT_ITEM_COL,
        item_feat_col=None,
        y_col=DEFAULT_RATING_COL,
    ):
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {it: i for i, it in enumerate(items)}

        self.user_ids = torch.tensor(
            [self.user2idx[u] for u in df[user_col]], dtype=torch.long
        )
        self.item_ids = torch.tensor(
            [self.item2idx[it] for it in df[item_col]], dtype=torch.long
        )

        if item_feat_col is not None and item_feat_col in df.columns:
            feats = df[item_feat_col].tolist()
            self.item_feats = torch.tensor(
                np.array(feats, dtype=np.float32), dtype=torch.float
            )
        else:
            self.item_feats = None

        if y_col is not None and y_col in df.columns:
            self.ratings = torch.tensor(
                df[y_col].values.astype(np.float32), dtype=torch.float
            )
        else:
            self.ratings = None

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        iid = self.item_ids[idx]
        feat = self.item_feats[idx] if self.item_feats is not None else torch.tensor([])
        if self.ratings is not None:
            return uid, iid, feat, self.ratings[idx]
        return uid, iid, feat


def _build_optimizer(name, params, lr=0.01, **kwargs):
    """Build a PyTorch optimizer by name.

    Default hyper-parameters are chosen to match TensorFlow v1 optimizers
    so that training dynamics are comparable.

    Args:
        name (str): One of ``"adagrad"``, ``"adadelta"``, ``"adam"``,
            ``"sgd"``, ``"rmsprop"``.
        params: Parameters to optimise.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer
    """
    name = name.lower()
    if name == "adagrad":
        # TF default: initial_accumulator_value=0.1
        return torch.optim.Adagrad(params, lr=lr, initial_accumulator_value=0.1)
    elif name == "adadelta":
        # TF defaults: rho=0.95, epsilon=1e-8
        return torch.optim.Adadelta(params, lr=lr, rho=0.95, eps=1e-8)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=kwargs.get("momentum", 0.0))
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=kwargs.get("momentum", 0.0))
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
