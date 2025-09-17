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

import pandas as pd
import pyarrow as pa

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


def list_(value_type: Union[pa.DataType, pa.Field]):
    """
    Create ``pd.ArrowDtype(pa.ListType)`` instance from a list or field.
    """
    if ArrowDtype is None:
        raise ImportError("ArrowDtype is not supported in current environment")
    return pd.ArrowDtype(pa.list_(value_type))


def struct_(fields: Union[pa.Schema, list]):
    """
    Create ``pd.ArrowDtype(pa.StructType)`` instance from fields.

    Parameters
    ----------
    fields : pyarrow.Schema or list of pyarrow.Field
        The fields to create the struct type from.

    Returns
    -------
    ArrowDtype
        An ArrowDtype instance with a StructType.
    """
    if ArrowDtype is None:
        raise ImportError("ArrowDtype is not supported in current environment")
    return pd.ArrowDtype(pa.struct(fields))


def is_map_dtype(dtype: ArrowDtype) -> bool:
    """
    Check whether the dtype is a map type.
    """
    if ArrowDtype is None:
        raise ImportError("ArrowDtype is not supported in current environment")
    return isinstance(dtype, ArrowDtype) and isinstance(dtype.pyarrow_dtype, pa.MapType)


def is_list_dtype(dtype: ArrowDtype) -> bool:
    """
    Check whether the dtype is a list dtype.
    """
    if ArrowDtype is None:
        raise ImportError("ArrowDtype is not supported in current environment")
    return isinstance(dtype, ArrowDtype) and isinstance(
        dtype.pyarrow_dtype, pa.ListType
    )


def is_struct_dtype(dtype: ArrowDtype) -> bool:
    """
    Check whether the dtype is a struct dtype.

    Parameters
    ----------
    dtype : ArrowDtype
        The dtype to check.

    Returns
    -------
    bool
        True if the dtype is a struct type, False otherwise.
    """
    if ArrowDtype is None:
        raise ImportError("ArrowDtype is not supported in current environment")
    return isinstance(dtype, ArrowDtype) and isinstance(
        dtype.pyarrow_dtype, pa.StructType
    )
