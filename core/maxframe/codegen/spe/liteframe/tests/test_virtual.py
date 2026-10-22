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

import pandas as pd
import pyarrow as pa
import pytest

from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.liteframe.virtual import LiteFrameFetchAdapter
from maxframe.liteframe.operators.virtual import LiteFrameFetch
from maxframe.utils.datatypes import is_arrow_dtype_supported

_requires_arrow_dtype = pytest.mark.skipif(
    not is_arrow_dtype_supported(),
    reason="pandas ArrowDtype is not supported in current environment",
)


@_requires_arrow_dtype
def test_fetch_adapter_returns_empty():
    """LiteFrameFetchAdapter should return empty code list (registration only)."""
    adapter = LiteFrameFetchAdapter()
    context = SPECodeContext()
    # Create a minimal fetch op for testing
    op = LiteFrameFetch(
        source_key="test_key",
        dtypes=pd.Series([pd.ArrowDtype(pa.int64())], index=["a"]),
    )
    result = adapter.generate_code(op, context)
    assert result == []
