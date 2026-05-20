# Copyright 1999-2026 Alibaba Group Holding Ltd.
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
import pyarrow as pa
import pytest
from pandas.api.types import is_string_dtype

from maxframe.core import OutputType
from maxframe.dataframe.typing_ import dtype, get_function_output_meta
from maxframe.lib.dtypes_extension import ArrowDtype, ExternalBlobDtype


def test_dataframe_type_annotation():
    def func() -> pd.DataFrame[int]:
        pass

    meta = get_function_output_meta(func)
    assert meta is not None
    assert meta.output_type.name == "dataframe"
    assert len(meta.dtypes) == 1
    # Use iloc for position-based access as index dtype may be StringDtype in pandas 3.0
    assert meta.dtypes.iloc[0] == np.dtype(int)

    def func1() -> pd.DataFrame[{"col1": int, "col2": float}]:  # noqa: F821
        pass

    def func2() -> pd.DataFrame["col1":int, "col2":float]:  # noqa: F821
        pass

    for func in [func1, func2]:
        meta = get_function_output_meta(func)
        assert meta is not None
        assert meta.output_type.name == "dataframe"
        assert len(meta.dtypes) == 2
        # Use iloc for position-based access as index dtype may be StringDtype in pandas 3.0
        assert meta.dtypes.iloc[0] == np.dtype(int)
        assert meta.dtypes.iloc[1] == np.dtype(float)

    def func() -> pd.DataFrame[str, {"col1": int, "col2": float}]:  # noqa: F821
        pass

    meta = get_function_output_meta(func)
    assert meta is not None
    assert meta.output_type.name == "dataframe"
    assert len(meta.dtypes) == 2

    meta_dtype = meta.index_value.value.dtype
    assert is_string_dtype(meta_dtype)

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
    meta_dtype = meta.index_value.value.dtype
    assert is_string_dtype(meta_dtype)


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

        meta_dtypes = meta.index_value.value.dtypes
        assert is_string_dtype(meta_dtypes.iloc[0])
        assert meta_dtypes.iloc[1] == np.dtype("int64")


def test_function_output_meta_corner_cases():
    def func():
        pass

    assert get_function_output_meta(func) is None
    assert get_function_output_meta("non-func-obj") is None

    def func() -> int:
        pass

    meta = get_function_output_meta(func)
    assert meta.dtype == np.dtype("int64")


def test_dtype_from_string():
    # Test basic type strings
    result = dtype("int64")
    assert isinstance(result, ArrowDtype)
    assert result.pyarrow_dtype == pa.int64()

    result = dtype("string")
    # wrap_arrow_dtype returns pd.StringDtype("pyarrow") for string types
    assert isinstance(result, (ArrowDtype, pd.StringDtype))
    if isinstance(result, ArrowDtype):
        assert result.pyarrow_dtype == pa.string()

    result = dtype("float32")
    assert isinstance(result, ArrowDtype)
    assert result.pyarrow_dtype == pa.float32()

    # Test complex type strings
    result = dtype("list<item: int64>")
    assert isinstance(result, ArrowDtype)
    assert pa.types.is_list(result.pyarrow_dtype)
    assert result.pyarrow_dtype.value_type == pa.int64()

    # Test blob type
    result = dtype("blob")
    assert isinstance(result, ExternalBlobDtype)

    result = dtype("BLOB")
    assert isinstance(result, ExternalBlobDtype)


def test_dtype_from_numpy():
    # Test NumPy dtype conversion
    result = dtype(np.dtype("int64"))
    assert isinstance(result, ArrowDtype)
    assert result.pyarrow_dtype == pa.int64()

    result = dtype(np.dtype("float32"))
    assert isinstance(result, ArrowDtype)
    assert result.pyarrow_dtype == pa.float32()

    result = dtype(np.dtype("int8"))
    assert isinstance(result, ArrowDtype)
    assert result.pyarrow_dtype == pa.int8()


def test_dtype_from_pandas_extension():
    # Test ArrowDtype input (should pass through)
    arrow_dtype = ArrowDtype(pa.int64())
    result = dtype(arrow_dtype)
    assert result is arrow_dtype

    # Test ExternalBlobDtype input (should pass through)
    blob_dtype = ExternalBlobDtype()
    result = dtype(blob_dtype)
    assert result is blob_dtype

    # Test pandas StringDtype (should convert to pyarrow-backed StringDtype)
    string_dtype = pd.StringDtype()
    result = dtype(string_dtype)
    assert isinstance(result, pd.StringDtype) or isinstance(result, ArrowDtype)
    assert isinstance(result, ArrowDtype) or result.storage == "pyarrow"

    # Test pandas StringDtype("pyarrow") (should return same type)
    try:
        string_dtype_pyarrow = pd.StringDtype("pyarrow")
        result = dtype(string_dtype_pyarrow)
        assert isinstance(result, pd.StringDtype) or isinstance(result, ArrowDtype)
        assert isinstance(result, ArrowDtype) or result.storage == "pyarrow"
    except ImportError:
        pass


def test_dtype_from_odps():
    try:
        from odps import types as odps_types
    except ImportError:
        pytest.skip("odps not installed")

    # Test ODPS basic types
    result = dtype(odps_types.string)
    # wrap_arrow_dtype returns pd.StringDtype("pyarrow") for string types
    assert isinstance(result, (ArrowDtype, pd.StringDtype))
    if isinstance(result, ArrowDtype):
        assert result.pyarrow_dtype == pa.string()

    result = dtype(odps_types.bigint)
    assert isinstance(result, ArrowDtype)
    assert result.pyarrow_dtype == pa.int64()

    result = dtype(odps_types.double)
    assert isinstance(result, ArrowDtype)
    assert result.pyarrow_dtype == pa.float64()

    # Test ODPS blob type if available
    if hasattr(odps_types, "blob"):
        result = dtype(odps_types.blob)
        assert isinstance(result, ExternalBlobDtype)

    # Test ODPS complex types
    result = dtype(odps_types.Array(odps_types.bigint))
    assert isinstance(result, ArrowDtype)
    assert pa.types.is_list(result.pyarrow_dtype)

    result = dtype(odps_types.Map(odps_types.string, odps_types.bigint))
    assert isinstance(result, ArrowDtype)
    assert pa.types.is_map(result.pyarrow_dtype)


def test_dtype_from_pyarrow():
    # Test PyArrow type input
    result = dtype(pa.int64())
    assert isinstance(result, ArrowDtype)
    assert result.pyarrow_dtype == pa.int64()

    result = dtype(pa.string())
    # wrap_arrow_dtype returns pd.StringDtype("pyarrow") for string types
    assert isinstance(result, (ArrowDtype, pd.StringDtype))
    if isinstance(result, ArrowDtype):
        assert result.pyarrow_dtype == pa.string()

    result = dtype(pa.list_(pa.int64()))
    assert isinstance(result, ArrowDtype)
    assert pa.types.is_list(result.pyarrow_dtype)

    result = dtype(pa.struct([("field1", pa.int64()), ("field2", pa.string())]))
    assert isinstance(result, ArrowDtype)
    assert pa.types.is_struct(result.pyarrow_dtype)


@pytest.mark.parametrize(
    "type_class,expected_type,expected_value",
    [
        (int, ArrowDtype, pa.int64()),
        (float, ArrowDtype, pa.float64()),
        (bool, ArrowDtype, pa.bool_()),
        (str, pd.StringDtype, "pyarrow"),
        (bytes, ArrowDtype, pa.binary()),
    ],
)
def test_dtype_from_python_type_classes(type_class, expected_type, expected_value):
    try:
        result = dtype(type_class)
    except ImportError as ex:
        if "pyarrow" in str(ex):
            pytest.skip("pyarrow ImportError: " + str(ex))
        else:
            raise

    assert isinstance(result, expected_type)
    if expected_type == pd.StringDtype:
        assert result.storage == expected_value
    else:
        assert result.pyarrow_dtype == expected_value


def test_dtype_errors():
    # Test unsupported types
    with pytest.raises(TypeError, match="Unsupported type"):
        dtype(12345)  # int is not a valid dtype input

    with pytest.raises(TypeError, match="Unsupported type"):
        dtype(["int64"])  # list is not a valid dtype input

    with pytest.raises(TypeError, match="Unsupported type"):
        dtype({"type": "int64"})  # dict is not a valid dtype input

    # Test invalid string
    with pytest.raises((ValueError, TypeError)):
        dtype("invalid_type_that_does_not_exist")


def test_series_type_dict_annotation():
    """Test parsing dict-based element type annotations"""
    from maxframe.dataframe.typing_ import SeriesType

    st = SeriesType.from_getitem_args({"col1": int, "col2": str})
    assert st.element_fields is not None
    assert len(st.element_fields) == 2
    assert st.element_fields[0].name == "col1"
    assert st.element_fields[0].dtype == np.dtype(int)
    assert st.element_fields[1].name == "col2"
    # str dtype may be StringDtype or object depending on version
    assert st.element_fields[1].dtype in [
        np.dtype(str),
        np.dtype("O"),
        pd.StringDtype(),
    ]


def test_get_function_output_meta_series_dict_axis1():
    """Test Series dict annotation with axis=1 produces DataFrame"""

    def func() -> pd.Series[{"col1": int, "col2": float}]:  # noqa: F821
        pass

    meta = get_function_output_meta(func, axis=1)
    assert meta is not None
    assert meta.output_type == OutputType.dataframe
    assert len(meta.dtypes) == 2
    assert list(meta.dtypes.index) == ["col1", "col2"]
    assert meta.dtypes["col1"] == np.dtype(int)
    assert meta.dtypes["col2"] == np.dtype(float)
