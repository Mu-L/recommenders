# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import pytest
import numpy as np
import pandas as pd

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    SEED,
)

try:
    from recommenders.utils.tf_utils import (
        build_optimizer,
        pandas_input_fn,
    )
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()  # need to disable eager in TF2.x
except ImportError:
    pass  # skip this import if we are in cpu environment


@pytest.fixture(scope="module")
def pd_df():
    df = pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2],
            DEFAULT_ITEM_COL: [1, 2, 3, 1, 4, 5],
            DEFAULT_RATING_COL: [5, 4, 3, 5, 5, 3],
        }
    )
    return df


@pytest.mark.gpu
def test_pandas_input_fn(pd_df):
    # check dataset
    dataset = pandas_input_fn(pd_df)()
    batch = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    with tf.compat.v1.Session() as sess:
        features = sess.run(batch)

        # check the input function returns all the columns
        assert len(features) == len(pd_df.columns)

        for k, v in features.items():
            assert k in pd_df.columns.values
            # check if a list feature column converted correctly
            if len(v.shape) == 1:
                assert np.array_equal(v, pd_df[k].values)
            elif len(v.shape) == 2:
                assert v.shape[1] == len(pd_df[k][0])

    # check dataset with shuffles
    dataset = pandas_input_fn(pd_df, shuffle=True, seed=SEED)()
    batch = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    with tf.compat.v1.Session() as sess:
        features = sess.run(batch)

        # check the input function returns all the columns
        assert len(features) == len(pd_df.columns)

        for k, v in features.items():
            assert k in pd_df.columns.values
            # check if a list feature column converted correctly
            if len(v.shape) == 1:
                assert not np.array_equal(v, pd_df[k].values)
            elif len(v.shape) == 2:
                assert v.shape[1] == len(pd_df[k][0])

    # check dataset w/ label
    dataset_with_label = pandas_input_fn(pd_df, y_col=DEFAULT_RATING_COL)()
    batch = tf.compat.v1.data.make_one_shot_iterator(dataset_with_label).get_next()
    with tf.compat.v1.Session() as sess:
        features, _ = sess.run(batch)
        assert (
            len(features) == len(pd_df.columns) - 1
        )  # label should not be in the features


@pytest.mark.gpu
def test_build_optimizer():
    adadelta = build_optimizer("Adadelta")
    assert isinstance(adadelta, tf.compat.v1.train.AdadeltaOptimizer)

    adagrad = build_optimizer("Adagrad")
    assert isinstance(adagrad, tf.compat.v1.train.AdagradOptimizer)

    adam = build_optimizer("Adam")
    assert isinstance(adam, tf.compat.v1.train.AdamOptimizer)

    ftrl = build_optimizer("Ftrl", **{"l1_regularization_strength": 0.001})
    assert isinstance(ftrl, tf.compat.v1.train.FtrlOptimizer)

    momentum = build_optimizer("Momentum", **{"momentum": 0.5})
    assert isinstance(momentum, tf.compat.v1.train.MomentumOptimizer)

    rmsprop = build_optimizer("RMSProp")
    assert isinstance(rmsprop, tf.compat.v1.train.RMSPropOptimizer)

    sgd = build_optimizer("SGD")
    assert isinstance(sgd, tf.compat.v1.train.GradientDescentOptimizer)


