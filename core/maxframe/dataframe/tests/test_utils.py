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
import pandas as pd
import pytest

from ...udf import MarkedFunction, with_python_requirements, with_resources
from ..utils import pack_func_args


@pytest.fixture
def df1():
    return pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])


def test_pack_function(df1):
    # pack normal function
    @with_resources("a.zip")
    def keep(df):
        return df

    f = pack_func_args(df1, keep)
    assert f(df1).equals(df1)
    assert isinstance(f, MarkedFunction)
    assert f.resources == ["a.zip"]

    # pack with args
    @with_python_requirements("numpy")
    def add(a, b):
        return a + b

    f = pack_func_args(df1, add, 1)
    assert f(df1).equals(df1 + 1)
    assert isinstance(f, MarkedFunction)
    assert f.pythonpacks[0].requirements == ("numpy",)

    f = pack_func_args(df1, np.sum)
    assert f(df1).equals(np.sum(df1))

    @with_resources("a.txt")
    @with_python_requirements("pandas")
    def times_add(df, param, times):
        return df * times + param

    f = pack_func_args(df1, times_add, 5, 6)
    assert f(df1).equals(df1 * 6 + 5)
    assert isinstance(f, MarkedFunction)
    assert f.resources == ["a.txt"]
    assert f.pythonpacks[0].requirements == ("pandas",)
