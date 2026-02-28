# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import pytest
import pandas as pd
import numpy as np

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
)

try:
    import torch
    from recommenders.models.wide_deep.wide_deep_utils import WideDeepModel
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
def test_init(pd_df):
    _, users, items = pd_df

    # Wide-only model
    wide = WideDeepModel(users=users, items=items, model_type="wide", crossed_feat_dim=10)
    assert isinstance(wide, WideDeepModel)
    assert wide.model_type == "wide"
    assert len(wide.users) == 2
    assert len(wide.items) == 5
    assert hasattr(wide, "wide_user")
    assert hasattr(wide, "wide_item")
    assert hasattr(wide, "wide_cross")
    assert not hasattr(wide, "dnn")

    # Deep-only model
    deep = WideDeepModel(users=users, items=items, model_type="deep")
    assert deep.model_type == "deep"
    assert hasattr(deep, "deep_user")
    assert hasattr(deep, "deep_item")
    assert hasattr(deep, "dnn")
    assert not hasattr(deep, "wide_user")

    # Wide & Deep model
    wd = WideDeepModel(users=users, items=items, model_type="wide_deep")
    assert wd.model_type == "wide_deep"
    assert hasattr(wd, "wide_user")
    assert hasattr(wd, "deep_user")
    assert hasattr(wd, "dnn")

    # With item features
    wd_feat = WideDeepModel(
        users=users,
        items=items,
        model_type="wide_deep",
        item_feat_col=ITEM_FEAT_COL,
        item_feat_shape=3,
    )
    assert wd_feat.item_feat_shape == 3

    # Invalid model_type
    with pytest.raises(ValueError):
        WideDeepModel(users=users, items=items, model_type="invalid")


@pytest.mark.gpu
def test_wide_model(pd_df):
    data, users, items = pd_df

    model = WideDeepModel(
        users=users, items=items, model_type="wide", crossed_feat_dim=10
    )
    model.fit(data, n_epochs=1, batch_size=2)

    preds = model.predict(data)
    assert len(preds) == len(data)


@pytest.mark.gpu
def test_deep_model(pd_df):
    data, users, items = pd_df

    model = WideDeepModel(users=users, items=items, model_type="deep")
    model.fit(data, n_epochs=1, batch_size=2)

    preds = model.predict(data)
    assert len(preds) == len(data)


@pytest.mark.gpu
def test_wide_deep_model(pd_df):
    data, users, items = pd_df

    model = WideDeepModel(users=users, items=items, model_type="wide_deep")
    model.fit(data, n_epochs=1, batch_size=2)

    preds = model.predict(data)
    assert len(preds) == len(data)


@pytest.mark.gpu
def test_wide_deep_model_with_item_features(pd_df):
    data, users, items = pd_df

    model = WideDeepModel(
        users=users,
        items=items,
        model_type="wide_deep",
        item_feat_col=ITEM_FEAT_COL,
        item_feat_shape=3,
    )
    model.fit(data, n_epochs=1, batch_size=2)

    preds = model.predict(data)
    assert len(preds) == len(data)


@pytest.mark.gpu
def test_recommend_k_items(pd_df):
    data, users, items = pd_df

    model = WideDeepModel(
        users=users,
        items=items,
        model_type="wide_deep",
    )
    model.fit(data, n_epochs=1, batch_size=2)

    test_df = pd.DataFrame({DEFAULT_USER_COL: [1, 2]})
    top_k = model.recommend_k_items(test_df, top_k=3, remove_seen=True)

    assert DEFAULT_USER_COL in top_k.columns
    assert DEFAULT_ITEM_COL in top_k.columns
    assert DEFAULT_PREDICTION_COL in top_k.columns
    # Each user should have at most top_k recommendations
    for uid in [1, 2]:
        user_recs = top_k[top_k[DEFAULT_USER_COL] == uid]
        assert len(user_recs) <= 3
    # With remove_seen=True, recommended items should not be in training set
    seen = set(zip(data[DEFAULT_USER_COL], data[DEFAULT_ITEM_COL]))
    for _, row in top_k.iterrows():
        assert (row[DEFAULT_USER_COL], row[DEFAULT_ITEM_COL]) not in seen


@pytest.mark.gpu
def test_recommend_k_items_with_item_features(pd_df):
    data, users, items = pd_df

    model = WideDeepModel(
        users=users,
        items=items,
        model_type="wide_deep",
        item_feat_col=ITEM_FEAT_COL,
        item_feat_shape=3,
    )
    model.fit(data, n_epochs=1, batch_size=2)

    test_df = pd.DataFrame({DEFAULT_USER_COL: [1, 2]})
    top_k = model.recommend_k_items(test_df, top_k=2, remove_seen=True)

    assert len(top_k) > 0
    assert DEFAULT_PREDICTION_COL in top_k.columns
    for uid in [1, 2]:
        user_recs = top_k[top_k[DEFAULT_USER_COL] == uid]
        assert len(user_recs) <= 2
