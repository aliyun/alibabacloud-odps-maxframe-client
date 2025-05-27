# Copyright 1999-2025 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools

import numpy as np
import pandas as pd
import pytest

from .... import dataframe as md
from .... import tensor as mt
from ....dataframe.core import DATAFRAME_TYPE
from .._split import KFold, train_test_split


def test_train_test_split_errors():
    pytest.raises(ValueError, train_test_split)

    pytest.raises(ValueError, train_test_split, range(3), train_size=1.1)

    pytest.raises(ValueError, train_test_split, range(3), test_size=0.6, train_size=0.6)
    pytest.raises(
        ValueError,
        train_test_split,
        range(3),
        test_size=np.float32(0.6),
        train_size=np.float32(0.6),
    )
    pytest.raises(ValueError, train_test_split, range(3), test_size="wrong_type")
    pytest.raises(ValueError, train_test_split, range(3), test_size=2, train_size=4)
    pytest.raises(TypeError, train_test_split, range(3), some_argument=1.1)
    pytest.raises(ValueError, train_test_split, range(3), range(42))
    pytest.raises(ValueError, train_test_split, range(10), shuffle=False, stratify=True)

    with pytest.raises(
        ValueError,
        match=r"train_size=11 should be either positive and "
        r"smaller than the number of samples 10 or a "
        r"float in the \(0, 1\) range",
    ):
        train_test_split(range(10), train_size=11, test_size=1)


def test_train_test_split_invalid_sizes1():
    for train_size, test_size in [
        (1.2, 0.8),
        (1.0, 0.8),
        (0.0, 0.8),
        (-0.2, 0.8),
        (0.8, 1.2),
        (0.8, 1.0),
        (0.8, 0.0),
        (0.8, -0.2),
    ]:
        with pytest.raises(ValueError, match=r"should be .* in the \(0, 1\) range"):
            train_test_split(range(10), train_size=train_size, test_size=test_size)


def test_train_test_split_invalid_sizes2():
    for train_size, test_size in [
        (-10, 0.8),
        (0, 0.8),
        (11, 0.8),
        (0.8, -10),
        (0.8, 0),
        (0.8, 11),
    ]:
        with pytest.raises(ValueError, match=r"should be .* in the \(0, 1\) range"):
            train_test_split(range(10), train_size=train_size, test_size=test_size)


def test_train_test_split():
    # allow nd-arrays
    X_4d = np.arange(10 * 5 * 3 * 2).reshape(10, 5, 3, 2)
    y_3d = np.arange(10 * 7 * 11).reshape(10, 7, 11)
    split = train_test_split(X_4d, y_3d)
    assert split[0].shape == (7, 5, 3, 2)
    assert split[1].shape == (3, 5, 3, 2)
    assert split[2].shape == (7, 7, 11)
    assert split[3].shape == (3, 7, 11)


def test_train_test_split_dataframe():
    X = np.ones(10)
    types = [pd.DataFrame, md.DataFrame]
    for InputFeatureType in types:
        # X dataframe
        X_df = InputFeatureType(X)
        X_train, X_test = train_test_split(X_df)
        assert isinstance(X_train, DATAFRAME_TYPE)
        assert isinstance(X_test, DATAFRAME_TYPE)


def test_mixed_input_type_train_test_split():
    rs = np.random.RandomState(0)
    df_raw = pd.DataFrame(rs.rand(10, 4))
    df = md.DataFrame(df_raw, chunk_size=5)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    for x_to_tensor, y_to_tensor in itertools.product(range(1), range(1)):
        x = X
        if x_to_tensor:
            x = mt.tensor(x)
        yy = y
        if y_to_tensor:
            yy = mt.tensor(yy)

        x_train, x_test, y_train, y_test = train_test_split(x, y)
        assert isinstance(x_train, type(x))
        assert isinstance(x_test, type(x))
        assert isinstance(y_train, type(yy))
        assert isinstance(y_test, type(yy))


def test_kfold_valueerrors():
    X1 = np.array([[1, 2], [3, 4], [5, 6]])
    # Check that errors are raised if there is not enough samples
    with pytest.raises(ValueError):
        next(KFold(4).split(X1))

    # Error when number of folds is <= 1
    with pytest.raises(ValueError):
        KFold(0)
    with pytest.raises(ValueError):
        KFold(1)

    # When n_splits is not integer:
    with pytest.raises(ValueError):
        KFold(1.5)
    with pytest.raises(ValueError):
        KFold(2.0)

    # When shuffle is not  a bool:
    with pytest.raises(TypeError):
        KFold(n_splits=4, shuffle=None)


def test_kfold_balance():
    # Check that KFold returns folds with balanced sizes
    for i in range(11, 17):
        kf = KFold(5).split(X=np.ones(i))
        sizes = [len(test) for _, test in kf]

        assert (np.max(sizes) - np.min(sizes)) <= 1
        assert np.sum(sizes) == i
