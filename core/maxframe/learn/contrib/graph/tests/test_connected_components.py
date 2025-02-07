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

import numpy as np
import pytest

from ..... import dataframe as md
from .....dataframe.core import DataFrameData
from .....tensor.core import TensorData
from .. import connected_components


@pytest.fixture
def df1():
    return md.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})


@pytest.fixture
def df2():
    return md.DataFrame(
        [[1, "2"], [1, "2"]],
        columns=["a", "b"],
    )


def test_connected_components(df1, df2):
    edges, flag = connected_components(df1, "a", "b")
    assert edges.op.max_iter == 6
    assert edges.shape == (np.nan, 2)
    assert isinstance(edges.data, DataFrameData)
    assert isinstance(flag.data, TensorData)
    assert flag.shape == ()
    assert "id" in edges.dtypes and "component" in edges.dtypes

    with pytest.raises(ValueError):
        connected_components(df1, "a", "x")

    with pytest.raises(ValueError):
        connected_components(df1, "a", "b", 0)

    with pytest.raises(ValueError):
        connected_components(df2, "a", "b")
