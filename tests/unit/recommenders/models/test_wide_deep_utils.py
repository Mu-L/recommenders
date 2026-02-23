# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import pytest
import pandas as pd
import numpy as np

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
)

try:
    import torch
    from recommenders.models.wide_deep.wide_deep_utils import (
        WideDeepModel,
        build_model,
        build_feature_columns,
        train_model,
        predict,
    )
except ImportError:
    pass  # skip this import if we are in cpu environment


ITEM_FEAT_COL = "itemFeat"


@pytest.fixture(scope="module")
def pd_df():
    df = pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2],
            DEFAULT_ITEM_COL: [1, 2, 3, 1, 4, 5],
            ITEM_FEAT_COL: [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [1, 1, 1],
                [4, 4, 4],
                [5, 5, 5],
            ],
            DEFAULT_RATING_COL: [5, 4, 3, 5, 5, 3],
        }
    )
    users = df.drop_duplicates(DEFAULT_USER_COL)[DEFAULT_USER_COL].values
    items = df.drop_duplicates(DEFAULT_ITEM_COL)[DEFAULT_ITEM_COL].values
    return df, users, items


@pytest.mark.gpu
def test_wide_model(pd_df):
    data, users, items = pd_df

    # Build wide-only feature columns
    wide_columns, deep_columns = build_feature_columns(
        users, items, model_type="wide", crossed_feat_dim=10
    )
    # Wide config should be populated, deep should be empty
    assert len(wide_columns) == 7
    assert not deep_columns
    assert wide_columns["crossed_feat_dim"] == 10

    # Build model
    model = build_model(wide_columns=wide_columns)
    assert isinstance(model, WideDeepModel)
    assert model.model_type == "wide"
    assert hasattr(model, "wide_user")
    assert hasattr(model, "wide_item")
    assert hasattr(model, "wide_cross")
    assert not hasattr(model, "dnn")

    # Train for 1 step
    model = train_model(
        model,
        data,
        list(users),
        list(items),
        batch_size=2,
        steps=1,
    )

    # Predict
    preds = predict(model, data, list(users), list(items))
    assert len(preds) == len(data)


@pytest.mark.gpu
def test_deep_model(pd_df):
    data, users, items = pd_df

    # Build deep-only feature columns
    wide_columns, deep_columns = build_feature_columns(
        users, items, model_type="deep"
    )
    assert not wide_columns
    assert deep_columns

    # Build model
    model = build_model(deep_columns=deep_columns)
    assert isinstance(model, WideDeepModel)
    assert model.model_type == "deep"
    assert hasattr(model, "deep_user")
    assert hasattr(model, "deep_item")
    assert hasattr(model, "dnn")
    assert not hasattr(model, "wide_user")

    # Train for a full pass over the data (1 epoch ≈ ceil(6/2) = 3 steps)
    model = train_model(
        model,
        data,
        list(users),
        list(items),
        batch_size=2,
        steps=3,
    )

    # Predict
    preds = predict(model, data, list(users), list(items))
    assert len(preds) == len(data)


@pytest.mark.gpu
def test_wide_deep_model(pd_df):
    data, users, items = pd_df

    # Build wide+deep feature columns
    wide_columns, deep_columns = build_feature_columns(
        users, items, model_type="wide_deep"
    )
    assert wide_columns
    assert deep_columns

    # Build model
    model = build_model(
        wide_columns=wide_columns,
        deep_columns=deep_columns,
    )
    assert isinstance(model, WideDeepModel)
    assert model.model_type == "wide_deep"
    assert hasattr(model, "wide_user")
    assert hasattr(model, "deep_user")
    assert hasattr(model, "dnn")

    # Train for 1 step
    model = train_model(
        model,
        data,
        list(users),
        list(items),
        batch_size=2,
        steps=1,
    )

    # Predict
    preds = predict(model, data, list(users), list(items))
    assert len(preds) == len(data)


@pytest.mark.gpu
def test_wide_deep_model_with_item_features(pd_df):
    data, users, items = pd_df

    # Build wide+deep feature columns with item features
    wide_columns, deep_columns = build_feature_columns(
        users,
        items,
        model_type="wide_deep",
        item_feat_col=ITEM_FEAT_COL,
        item_feat_shape=3,
    )
    assert deep_columns["item_feat_shape"] == 3

    # Build model
    model = build_model(
        wide_columns=wide_columns,
        deep_columns=deep_columns,
    )
    assert model.item_feat_shape == 3

    # Train for 1 step
    model = train_model(
        model,
        data,
        list(users),
        list(items),
        item_feat_col=ITEM_FEAT_COL,
        batch_size=2,
        steps=1,
    )

    # Predict
    preds = predict(
        model, data, list(users), list(items),
        item_feat_col=ITEM_FEAT_COL,
    )
    assert len(preds) == len(data)
