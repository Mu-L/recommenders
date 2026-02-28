# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import os
import pytest
import numpy as np
import pandas as pd

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
)

try:
    import torch
    import torch.nn as nn
    from recommenders.models.wide_deep.wide_deep_utils import (
        WideDeepModel,
        _build_optimizer,
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
def test_init(pd_df):
    _, users, items = pd_df

    # Wide-only model
    wide = WideDeepModel(users=users, items=items, model_type="wide", crossed_feat_dim=10)
    assert isinstance(wide, WideDeepModel)
    assert wide.model_type == "wide"
    assert wide.n_users == 2
    assert wide.n_items == 5
    assert len(wide.users) == 2
    assert len(wide.items) == 5
    assert wide.crossed_feat_dim == 10
    assert hasattr(wide, "wide_user")
    assert hasattr(wide, "wide_item")
    assert hasattr(wide, "wide_cross")
    assert not hasattr(wide, "dnn")
    assert torch.all(wide.wide_user.weight == 0)
    assert torch.all(wide.wide_item.weight == 0)
    assert torch.all(wide.wide_cross.weight == 0)

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

    # Custom column names
    custom = WideDeepModel(
        users=users, items=items, model_type="wide_deep",
        user_col="uid", item_col="iid",
    )
    assert custom.user_col == "uid"
    assert custom.item_col == "iid"

    # Custom embedding dims and DNN architecture
    custom_dnn = WideDeepModel(
        users=users, items=items, model_type="deep",
        user_dim=16, item_dim=4, dnn_hidden_units=(64,),
    )
    assert custom_dnn.deep_user.embedding_dim == 16
    assert custom_dnn.deep_item.embedding_dim == 4
    # Single hidden layer: Linear(20,64) -> ReLU -> BN -> Linear(64,1)
    layer_types = [type(m) for m in custom_dnn.dnn]
    assert layer_types == [nn.Linear, nn.ReLU, nn.BatchNorm1d, nn.Linear]
    assert custom_dnn.dnn[0].in_features == 20  # 16 + 4
    assert custom_dnn.dnn[0].out_features == 64

    # Dropout enabled
    with_dropout = WideDeepModel(
        users=users, items=items, model_type="deep", dnn_dropout=0.5,
    )
    layer_types = [type(m) for m in with_dropout.dnn]
    assert nn.Dropout in layer_types

    # Batch norm disabled
    no_bn = WideDeepModel(
        users=users, items=items, model_type="deep", dnn_batch_norm=False,
    )
    layer_types = [type(m) for m in no_bn.dnn]
    assert nn.BatchNorm1d not in layer_types

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
    assert all(np.isfinite(preds))


@pytest.mark.gpu
def test_deep_model(pd_df):
    data, users, items = pd_df

    model = WideDeepModel(users=users, items=items, model_type="deep")
    model.fit(data, n_epochs=1, batch_size=2)

    preds = model.predict(data)
    assert len(preds) == len(data)
    assert all(np.isfinite(preds))


@pytest.mark.gpu
def test_wide_deep_model(pd_df):
    data, users, items = pd_df

    model = WideDeepModel(users=users, items=items, model_type="wide_deep")
    model.fit(data, n_epochs=1, batch_size=2)

    preds = model.predict(data)
    assert len(preds) == len(data)
    assert all(np.isfinite(preds))


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
    assert all(np.isfinite(preds))


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

    # recommend_k_items before fit() should raise RuntimeError
    unfitted = WideDeepModel(users=users, items=items, model_type="wide_deep")
    with pytest.raises(RuntimeError):
        unfitted.recommend_k_items(test_df, top_k=3, remove_seen=True)
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

    # remove_seen=False should include seen items
    top_k_all = model.recommend_k_items(test_df, top_k=3, remove_seen=False)
    assert len(top_k_all) > 0
    for uid in [1, 2]:
        user_recs = top_k_all[top_k_all[DEFAULT_USER_COL] == uid]
        assert len(user_recs) <= 3


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
    assert DEFAULT_USER_COL in top_k.columns
    assert DEFAULT_ITEM_COL in top_k.columns
    assert DEFAULT_PREDICTION_COL in top_k.columns
    for uid in [1, 2]:
        user_recs = top_k[top_k[DEFAULT_USER_COL] == uid]
        assert len(user_recs) <= 2
    # With remove_seen=True, recommended items should not be in training set
    seen = set(zip(data[DEFAULT_USER_COL], data[DEFAULT_ITEM_COL]))
    for _, row in top_k.iterrows():
        assert (row[DEFAULT_USER_COL], row[DEFAULT_ITEM_COL]) not in seen

    # remove_seen=False should include seen items
    top_k_all = model.recommend_k_items(test_df, top_k=2, remove_seen=False)
    assert len(top_k_all) > 0
    for uid in [1, 2]:
        user_recs = top_k_all[top_k_all[DEFAULT_USER_COL] == uid]
        assert len(user_recs) <= 2


@pytest.mark.gpu
@pytest.mark.parametrize("optimizer", ["adam", "sgd", "rmsprop"])
def test_optimizers(pd_df, optimizer):
    data, users, items = pd_df

    model = WideDeepModel(users=users, items=items, model_type="wide_deep")
    model.fit(
        data,
        n_epochs=1,
        batch_size=2,
        wide_optimizer=optimizer,
        deep_optimizer=optimizer,
    )
    preds = model.predict(data)
    assert len(preds) == len(data)
    assert all(np.isfinite(preds))


@pytest.mark.gpu
def test_invalid_optimizer():
    dummy_params = [torch.nn.Parameter(torch.zeros(1))]
    with pytest.raises(ValueError):
        _build_optimizer("bad_name", dummy_params)


@pytest.mark.gpu
def test_seed_reproducibility(pd_df):
    data, users, items = pd_df

    # Constructor seed produces identical weights
    m1 = WideDeepModel(users=users, items=items, model_type="wide_deep", seed=42)
    m2 = WideDeepModel(users=users, items=items, model_type="wide_deep", seed=42)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)

    # fit() seed produces identical predictions
    m1.fit(data, n_epochs=3, batch_size=2, seed=99)
    m2.fit(data, n_epochs=3, batch_size=2, seed=99)
    preds1 = m1.predict(data)
    preds2 = m2.predict(data)
    assert np.allclose(preds1, preds2)


@pytest.mark.gpu
def test_save_load(pd_df, tmp):
    data, users, items = pd_df

    model = WideDeepModel(users=users, items=items, model_type="wide_deep", seed=42)
    model.fit(data, n_epochs=3, batch_size=2, seed=99)
    preds_before = model.predict(data)

    path = os.path.join(tmp, "wide_deep.pt")
    torch.save(model.state_dict(), path)

    loaded = WideDeepModel(users=users, items=items, model_type="wide_deep")
    loaded.load_state_dict(torch.load(path))
    preds_after = loaded.predict(data)

    assert np.allclose(preds_before, preds_after)


@pytest.mark.gpu
def test_cross_hash(pd_df):
    _, users, items = pd_df

    model = WideDeepModel(users=users, items=items, model_type="wide", crossed_feat_dim=100)
    uids = torch.tensor([0, 1, 0, 1])
    iids = torch.tensor([0, 1, 2, 0])

    hashes = model._cross_hash(uids, iids, 100)

    # All values within [0, bucket_size)
    assert (hashes >= 0).all()
    assert (hashes < 100).all()
    # Deterministic: same inputs produce same outputs
    assert torch.equal(hashes, model._cross_hash(uids, iids, 100))
    # Different (user, item) pairs produce different hashes
    assert len(hashes.unique()) == len(hashes)


@pytest.mark.gpu
def test_unknown_ids(pd_df):
    data, users, items = pd_df

    model = WideDeepModel(users=users, items=items, model_type="wide_deep")
    model.fit(data, n_epochs=1, batch_size=2)

    # Unknown user ID
    bad_user_df = pd.DataFrame(
        {DEFAULT_USER_COL: [999], DEFAULT_ITEM_COL: [1], DEFAULT_RATING_COL: [3.0]}
    )
    with pytest.raises(ValueError, match="Unknown user IDs"):
        model.predict(bad_user_df)

    # Unknown item ID
    bad_item_df = pd.DataFrame(
        {DEFAULT_USER_COL: [1], DEFAULT_ITEM_COL: [999], DEFAULT_RATING_COL: [3.0]}
    )
    with pytest.raises(ValueError, match="Unknown item IDs"):
        model.predict(bad_item_df)
