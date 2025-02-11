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
from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import ExtensionDtype

try:
    from pandas import ArrowDtype
except ImportError:
    try:
        from ._fake_arrow_dtype import FakeArrowDtype as ArrowDtype
    except ImportError:
        ArrowDtype = None


def dict_(
    key_type: Union[pa.DataType, pa.Field], item_type: Union[pa.DataType, pa.Field]
) -> ArrowDtype:
    """
    Create ``pd.ArrowDtype(pa.MapType)`` instance from key and item data types or fields.
    """
    if ArrowDtype is None:
        raise ImportError("ArrowDtype is not supported in current environment")
    return pd.ArrowDtype(pa.map_(key_type, item_type))


def is_map_dtype(dtype: ArrowDtype) -> bool:
    """
    Check whether the dtype is a map type.
    """
    if ArrowDtype is None:
        raise ImportError("ArrowDtype is not supported in current environment")
    return isinstance(dtype, ArrowDtype) and isinstance(dtype.pyarrow_dtype, pa.MapType)


_dtype_mapping = {
    pd.Int8Dtype(): lambda x: pa.int8(),
    pd.Int16Dtype(): lambda x: pa.int16(),
    pd.Int32Dtype(): lambda x: pa.int32(),
    pd.Int64Dtype(): lambda x: pa.int64(),
    pd.UInt8Dtype(): lambda x: pa.uint8(),
    pd.UInt16Dtype(): lambda x: pa.uint16(),
    pd.UInt32Dtype(): lambda x: pa.uint32(),
    pd.UInt64Dtype(): lambda x: pa.uint64(),
    pd.BooleanDtype(): lambda x: pa.bool_(),
    pd.Float32Dtype(): lambda x: pa.float32(),
    pd.Float64Dtype(): lambda x: pa.float64(),
    pd.StringDtype(): lambda x: pa.string(),
}


def infer_arrow_dtype(
    dtype: Union[np.dtype, pa.DataType, ExtensionDtype]
) -> Union[ArrowDtype, ExtensionDtype]:
    """
    Convert any pandas accepted dtype to arrow type in a best effort way.

    Parameters
    ----------
    dtype : Union[np.dtype, pa.DataType, ExtensionDtype]
        The dtype instance, can be np.dtype, pa.DataType or ExtensionDtype

    Returns
    -------
    Union[pd.ArrowDtype, ExtensionDtype]: The converted pd.ArrowDtype, or ExtensionDtype if conversion failed.
    """
    if isinstance(dtype, ArrowDtype):
        return dtype

    if isinstance(dtype, np.dtype):
        return ArrowDtype(pa.from_numpy_dtype(dtype))
    if isinstance(dtype, pd.DatetimeTZDtype):
        return pa.timestamp(dtype.unit, dtype.tz)

    if dtype in _dtype_mapping:
        return ArrowDtype(_dtype_mapping[dtype](dtype))
    return dtype
