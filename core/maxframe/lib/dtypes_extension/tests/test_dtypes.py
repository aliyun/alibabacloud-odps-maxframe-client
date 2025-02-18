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

import pandas as pd
import pyarrow as pa
import pytest

from ....utils import ARROW_DTYPE_NOT_SUPPORTED
from ..dtypes import dict_, is_map_dtype

try:
    from pandas import ArrowDtype
except:
    ArrowDtype = None


@pytest.mark.skipif(
    ARROW_DTYPE_NOT_SUPPORTED,
    reason="pandas doesn't support ArrowDtype",
)
def test_map_dtype():
    dt = dict_(pa.int64(), pa.string())
    assert is_map_dtype(dt)

    dt = pd.ArrowDtype(pa.list_(pa.int64()))
    assert not is_map_dtype(dt)
    assert not is_map_dtype(pd.Int64Dtype)
