# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import math
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
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
        n_users (int): Number of unique users.
        n_items (int): Number of unique items.
        model_type (str): ``"wide"``, ``"deep"``, or ``"wide_deep"``.
        crossed_feat_dim (int): Hash bucket size for the crossed user×item feature (wide).
        user_dim (int): User embedding dimension (deep).
        item_dim (int): Item embedding dimension (deep).
        item_feat_shape (int or None): Dimension of item feature vector.
            ``None`` means no item features.
        dnn_hidden_units (tuple of int): Hidden layer sizes for the deep component.
        dnn_dropout (float): Dropout rate for the deep component.
        dnn_batch_norm (bool): Whether to use batch normalization in the deep component.
    """

    def __init__(
        self,
        n_users,
        n_items,
        model_type="wide_deep",
        crossed_feat_dim=1000,
        user_dim=8,
        item_dim=8,
        item_feat_shape=None,
        dnn_hidden_units=(128, 128),
        dnn_dropout=0.0,
        dnn_batch_norm=True,
    ):
        super().__init__()

        if model_type not in ("wide", "deep", "wide_deep"):
            raise ValueError("model_type must be 'wide', 'deep', or 'wide_deep'")

        self.model_type = model_type
        self.n_users = n_users
        self.n_items = n_items
        self.crossed_feat_dim = crossed_feat_dim

        # --- Wide component ---
        if model_type in ("wide", "wide_deep"):
            self.wide_user = nn.Embedding(n_users, 1)
            self.wide_item = nn.Embedding(n_items, 1)
            self.wide_cross = nn.Embedding(crossed_feat_dim, 1)
            self.wide_bias = nn.Parameter(torch.zeros(1))
            nn.init.zeros_(self.wide_user.weight)
            nn.init.zeros_(self.wide_item.weight)
            nn.init.zeros_(self.wide_cross.weight)

        # --- Deep component ---
        if model_type in ("deep", "wide_deep"):
            self.deep_user = nn.Embedding(
                n_users, user_dim, max_norm=user_dim**0.5
            )
            self.deep_item = nn.Embedding(
                n_items, item_dim, max_norm=item_dim**0.5
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


# ---------------------------------------------------------------------------
# Helper functions 
# ---------------------------------------------------------------------------


def build_feature_columns(
    users,
    items,
    user_col=DEFAULT_USER_COL,
    item_col=DEFAULT_ITEM_COL,
    item_feat_col=None,
    crossed_feat_dim=1000,
    user_dim=8,
    item_dim=8,
    item_feat_shape=None,
    model_type="wide_deep",
):
    """Build wide and/or deep feature configuration dictionaries.

    It returns plain dictionaries that are later passed to :func:`build_model`.

    Args:
        users (iterable): Distinct user IDs.
        items (iterable): Distinct item IDs.
        user_col (str): User column name.
        item_col (str): Item column name.
        item_feat_col (str or None): Item feature column name.
        crossed_feat_dim (int): Hash bucket size for the crossed feature.
        user_dim (int): User embedding dimension (deep).
        item_dim (int): Item embedding dimension (deep).
        item_feat_shape (int or None): Item feature vector length.
        model_type (str): ``"wide"``, ``"deep"``, or ``"wide_deep"``.

    Returns:
        tuple[dict, dict]: ``(wide_columns, deep_columns)`` configuration
        dictionaries.  Empty dict when the component is not used.
    """
    if model_type not in ("wide", "deep", "wide_deep"):
        raise ValueError("model_type must be 'wide', 'deep', or 'wide_deep'")

    users = list(users)
    items = list(items)

    wide_columns = {}
    deep_columns = {}

    if model_type in ("wide", "wide_deep"):
        wide_columns = {
            "n_users": len(users),
            "n_items": len(items),
            "users": users,
            "items": items,
            "user_col": user_col,
            "item_col": item_col,
            "crossed_feat_dim": crossed_feat_dim,
        }

    if model_type in ("deep", "wide_deep"):
        deep_columns = {
            "n_users": len(users),
            "n_items": len(items),
            "users": users,
            "items": items,
            "user_col": user_col,
            "item_col": item_col,
            "user_dim": user_dim,
            "item_dim": item_dim,
            "item_feat_col": item_feat_col,
            "item_feat_shape": item_feat_shape,
        }

    return wide_columns, deep_columns


def build_model(
    wide_columns=None,
    deep_columns=None,
    dnn_hidden_units=(128, 128),
    dnn_dropout=0.0,
    dnn_batch_norm=True,
    seed=None,
):
    """Build a :class:`WideDeepModel` from feature configuration dicts.

    Args:
        wide_columns (dict or None): Wide config from :func:`build_feature_columns`.
        deep_columns (dict or None): Deep config from :func:`build_feature_columns`.
        dnn_hidden_units (tuple of int): Hidden layer sizes.
        dnn_dropout (float): Dropout rate.
        dnn_batch_norm (bool): Batch normalization flag.
        seed (int or None): Random seed.

    Returns:
        WideDeepModel: The constructed model.
    """
    if wide_columns is None:
        wide_columns = {}
    if deep_columns is None:
        deep_columns = {}

    if seed is not None:
        torch.manual_seed(seed)

    has_wide = bool(wide_columns)
    has_deep = bool(deep_columns)

    if not has_wide and not has_deep:
        raise ValueError(
            "Provide wide_columns and/or deep_columns to build a model."
        )

    if has_wide and has_deep:
        model_type = "wide_deep"
    elif has_wide:
        model_type = "wide"
    else:
        model_type = "deep"

    # Merge user/item counts from whichever config is present
    cfg = wide_columns if has_wide else deep_columns
    n_users = cfg["n_users"]
    n_items = cfg["n_items"]
    crossed_feat_dim = wide_columns.get("crossed_feat_dim", 0)
    user_dim = deep_columns.get("user_dim", 0)
    item_dim = deep_columns.get("item_dim", 0)
    item_feat_shape = deep_columns.get("item_feat_shape", None)

    model = WideDeepModel(
        n_users=n_users,
        n_items=n_items,
        model_type=model_type,
        crossed_feat_dim=crossed_feat_dim,
        user_dim=user_dim,
        item_dim=item_dim,
        item_feat_shape=item_feat_shape,
        dnn_hidden_units=dnn_hidden_units,
        dnn_dropout=dnn_dropout,
        dnn_batch_norm=dnn_batch_norm,
    )
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WideDeepDataset(Dataset):
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


# ---------------------------------------------------------------------------
# Training & prediction utilities
# ---------------------------------------------------------------------------


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


def train_model(
    model,
    train_df,
    users,
    items,
    y_col=DEFAULT_RATING_COL,
    user_col=DEFAULT_USER_COL,
    item_col=DEFAULT_ITEM_COL,
    item_feat_col=None,
    batch_size=32,
    steps=50000,
    wide_optimizer="adagrad",
    wide_optimizer_lr=0.01,
    deep_optimizer="adadelta",
    deep_optimizer_lr=0.01,
    seed=None,
    eval_fn=None,
    eval_every_n_steps=None,
    log_every_n_steps=5000,
):
    """Train a :class:`WideDeepModel`.

    Args:
        model (WideDeepModel): Model to train.
        train_df (pd.DataFrame): Training data.
        users (list): User ID vocabulary.
        items (list): Item ID vocabulary.
        y_col (str): Label column name.
        user_col (str): User column name.
        item_col (str): Item column name.
        item_feat_col (str or None): Item feature column name.
        batch_size (int): Training batch size.
        steps (int): Number of training steps (batches).
        wide_optimizer (str): Optimizer name for wide parameters
            (e.g. ``"adagrad"``).
        wide_optimizer_lr (float): Learning rate for the wide optimizer.
        deep_optimizer (str): Optimizer name for deep parameters
            (e.g. ``"adadelta"``).
        deep_optimizer_lr (float): Learning rate for the deep optimizer.
        seed (int or None): Random seed.
        eval_fn (callable or None): ``eval_fn(model, step)`` called every
            *eval_every_n_steps* steps.
        eval_every_n_steps (int or None): Evaluation frequency.
        log_every_n_steps (int): How often to log training loss.

    Returns:
        WideDeepModel: Trained model.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Build optimizers *after* .to(device) so parameters are on the right device
    wide_opt = None
    deep_opt = None
    if model.model_type in ("wide", "wide_deep"):
        wide_params = list(model.wide_user.parameters()) + \
                      list(model.wide_item.parameters()) + \
                      list(model.wide_cross.parameters()) + \
                      [model.wide_bias]
        wide_opt = _build_optimizer(wide_optimizer, wide_params, lr=wide_optimizer_lr)
    if model.model_type in ("deep", "wide_deep"):
        deep_params = list(model.deep_user.parameters()) + \
                      list(model.deep_item.parameters()) + \
                      list(model.dnn.parameters())
        deep_opt = _build_optimizer(deep_optimizer, deep_params, lr=deep_optimizer_lr)

    # Dataset and DataLoader
    dataset = WideDeepDataset(
        train_df, users, items,
        user_col=user_col, item_col=item_col,
        item_feat_col=item_feat_col, y_col=y_col,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=0,
    )

    loss_fn = nn.MSELoss()
    model.train()

    step = 0
    while step < steps:
        # Create a fresh iterator each epoch so data is reshuffled
        for batch in loader:
            if step >= steps:
                break
            step += 1

            uid, iid, feat, rating = [b.to(device) for b in batch]
            item_feats = feat if feat.numel() > 0 else None

            # Forward
            pred = model(uid, iid, item_feats)
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

            if step % log_every_n_steps == 0:
                logger.info("Step %d/%d – loss = %.4f", step, steps, loss.item())

            if eval_fn is not None and eval_every_n_steps and step % eval_every_n_steps == 0:
                model.eval()
                eval_fn(model, step)
                model.train()

    return model


def predict(
    model,
    df,
    users,
    items,
    user_col=DEFAULT_USER_COL,
    item_col=DEFAULT_ITEM_COL,
    item_feat_col=None,
    batch_size=256,
):
    """Generate predictions for the given dataframe.

    Args:
        model (WideDeepModel): Trained model.
        df (pd.DataFrame): Data to predict on.
        users (list): User ID vocabulary (same as training).
        items (list): Item ID vocabulary (same as training).
        user_col (str): User column name.
        item_col (str): Item column name.
        item_feat_col (str or None): Item feature column name.
        batch_size (int): Batch size for prediction.

    Returns:
        list[float]: Predicted values.
    """
    device = next(model.parameters()).device
    model.eval()

    dataset = WideDeepDataset(
        df, users, items,
        user_col=user_col, item_col=item_col,
        item_feat_col=item_feat_col, y_col=None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for batch in loader:
            uid, iid, feat = [b.to(device) for b in batch]
            item_feats = feat if feat.numel() > 0 else None
            out = model(uid, iid, item_feats)
            preds.extend(out.cpu().tolist())

    return preds
