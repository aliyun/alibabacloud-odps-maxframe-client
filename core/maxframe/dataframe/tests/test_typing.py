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

from ...core import OutputType
from ..typing_ import get_function_output_meta


def test_dataframe_type_annotation():
    def func() -> pd.DataFrame[int]:
        pass

    meta = get_function_output_meta(func)
    assert meta is not None
    assert meta.output_type.name == "dataframe"
    assert len(meta.dtypes) == 1
    assert meta.dtypes[0] == np.dtype(int)

    def func1() -> pd.DataFrame[{"col1": int, "col2": float}]:  # noqa: F821
        pass

    def func2() -> pd.DataFrame["col1":int, "col2":float]:  # noqa: F821
        pass

    for func in [func1, func2]:
        meta = get_function_output_meta(func)
        assert meta is not None
        assert meta.output_type.name == "dataframe"
        assert len(meta.dtypes) == 2
        assert meta.dtypes[0] == np.dtype(int)
        assert meta.dtypes[1] == np.dtype(float)

    def func() -> pd.DataFrame[str, {"col1": int, "col2": float}]:  # noqa: F821
        pass

    meta = get_function_output_meta(func)
    assert meta is not None
    assert meta.output_type.name == "dataframe"
    assert len(meta.dtypes) == 2
    assert meta.index_value.value.dtype == np.dtype("O")
    assert list(meta.dtypes.index) == ["col1", "col2"]
    assert list(meta.dtypes) == [np.dtype(int), np.dtype(float)]


def test_series_type_annotation():
    def func() -> pd.Series[np.str_]:
        pass

    meta = get_function_output_meta(func)
    assert meta is not None
    assert meta.output_type == OutputType.series
    assert meta.dtype == np.dtype(np.str_)

    def func() -> pd.Series[("idx_name", str), ("series_name", np.int64)]:  # noqa: F821
        pass

    meta = get_function_output_meta(func)
    assert meta is not None
    assert meta.output_type == OutputType.series
    assert meta.name == "series_name"
    assert meta.dtype == np.dtype(np.int64)
    assert meta.index_value.value._name == "idx_name"
    assert meta.index_value.value.dtype == np.dtype("O")


def test_index_type_annotation():
    def func1() -> pd.Index[np.int64]:
        pass

    def func2() -> pd.Index["ix" : np.int64]:  # noqa: F821
        pass

    for func in [func1, func2]:
        meta = get_function_output_meta(func)
        assert meta is not None
        assert meta.output_type == OutputType.index
        assert meta.index_value.value.dtype == np.dtype("int64")
        if func is func2:
            assert meta.index_value.value._name == "ix"

    def func3() -> pd.Index["ix1":str, "ix2" : np.int64]:  # noqa: F821
        pass

    def func4() -> pd.Index[[("ix1", str), ("ix2", np.int64)]]:  # noqa: F821
        pass

    for func in [func3, func4]:
        meta = get_function_output_meta(func)
        assert meta is not None
        assert meta.output_type == OutputType.index
        assert meta.index_value.value.names == ["ix1", "ix2"]
        assert list(meta.index_value.value.dtypes) == [np.dtype("O"), np.dtype("int64")]


def test_function_output_meta_corner_cases():
    def func():
        pass

    assert get_function_output_meta(func) is None
    assert get_function_output_meta("non-func-obj") is None

    def func() -> int:
        pass

    meta = get_function_output_meta(func)
    assert meta.dtype == np.dtype("int64")
