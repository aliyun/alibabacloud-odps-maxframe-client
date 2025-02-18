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


def is_map_dtype(dtype: ArrowDtype) -> bool:
    """
    Check whether the dtype is a map type.
    """
    if ArrowDtype is None:
        raise ImportError("ArrowDtype is not supported in current environment")
    return isinstance(dtype, ArrowDtype) and isinstance(dtype.pyarrow_dtype, pa.MapType)
