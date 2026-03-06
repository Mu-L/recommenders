# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
from tempfile import TemporaryDirectory

try:
    from recommenders.models.lightgbm import LightGBMRanker
except ModuleNotFoundError:
    pass


N_TRAIN = 200
N_TEST = 50
N_FEATURES = 10
SEED = 42


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.RandomState(SEED)
    train_x = rng.randn(N_TRAIN, N_FEATURES).astype(np.float32)
    train_y = rng.randint(0, 2, N_TRAIN).reshape(-1, 1)
    valid_x = rng.randn(N_TEST, N_FEATURES).astype(np.float32)
    valid_y = rng.randint(0, 2, N_TEST).reshape(-1, 1)
    test_x = rng.randn(N_TEST, N_FEATURES).astype(np.float32)
    return train_x, train_y, valid_x, valid_y, test_x


@pytest.fixture(scope="module")
def trained_ranker(synthetic_data):
    train_x, train_y, valid_x, valid_y, _ = synthetic_data
    ranker = LightGBMRanker(
        params={"objective": "binary", "metric": "auc", "num_threads": 1},
        num_boost_round=10,
        early_stopping_rounds=5,
    )
    ranker.fit(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
    return ranker


def test_fit_returns_self(synthetic_data):
    train_x, train_y, _, _, _ = synthetic_data
    ranker = LightGBMRanker(
        params={"objective": "binary", "num_threads": 1}, num_boost_round=5
    )
    result = ranker.fit(train_x, train_y)
    assert result is ranker


def test_fit_without_validation(synthetic_data):
    train_x, train_y, _, _, _ = synthetic_data
    ranker = LightGBMRanker(
        params={"objective": "binary", "num_threads": 1}, num_boost_round=5
    )
    ranker.fit(train_x, train_y)
    assert ranker.model is not None


def test_predict_shape(trained_ranker, synthetic_data):
    _, _, _, _, test_x = synthetic_data
    scores = trained_ranker.predict(test_x)
    assert scores.shape == (N_TEST,)


def test_predict_scores_in_range(trained_ranker, synthetic_data):
    _, _, _, _, test_x = synthetic_data
    scores = trained_ranker.predict(test_x)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_predict_before_fit_raises():
    ranker = LightGBMRanker()
    rng = np.random.RandomState(SEED)
    x = rng.randn(10, 5)
    with pytest.raises(RuntimeError):
        ranker.predict(x)


def test_save_before_fit_raises():
    ranker = LightGBMRanker()
    with pytest.raises(RuntimeError):
        ranker.save("model.lgb")


def test_save_and_load(trained_ranker, synthetic_data):
    _, _, _, _, test_x = synthetic_data
    original_scores = trained_ranker.predict(test_x)
    with TemporaryDirectory() as tmp:
        import os
        path = os.path.join(tmp, "ranker.lgb")
        trained_ranker.save(path)
        loaded = LightGBMRanker.load(path)
        loaded_scores = loaded.predict(test_x)
    np.testing.assert_allclose(original_scores, loaded_scores)


def test_params_override_defaults():
    custom_params = {"num_leaves": 128, "learning_rate": 0.05}
    ranker = LightGBMRanker(params=custom_params)
    assert ranker.params["num_leaves"] == 128
    assert ranker.params["learning_rate"] == 0.05
    # DEFAULT_PARAMS keys not overridden should still be present
    assert "objective" in ranker.params
