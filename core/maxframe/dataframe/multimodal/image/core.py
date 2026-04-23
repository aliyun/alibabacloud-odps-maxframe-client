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

from typing import TYPE_CHECKING, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import pyarrow as pa

from .... import opcodes
from ....core import OutputType
from ....serialization.serializables import StringField
from ...operators import DataFrameOperator, DataFrameOperatorMixin

if TYPE_CHECKING:
    from ...core import Series

_INT_METHODS = ("width", "height", "size")
_STR_METHODS = ("mode", "format")


class SeriesImageMethods(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.SERIES_IMAGE_METHODS

    method = StringField("method", default=None)
    errors = StringField("errors", default="ignore")
    dtype_backend = StringField("dtype_backend", default=None)

    def __init__(self, output_types=None, **kw):
        output_types = output_types or [OutputType.series]
        kw["_output_types"] = kw.get("_output_types") or output_types
        super().__init__(**kw)

    def __call__(self, series: "Series") -> "Series":
        dtype = self._resolve_dtype()
        return self.new_series(
            [series],
            shape=series.shape,
            dtype=dtype,
            index_value=series.index_value,
            name=series.name,
        )

    def _resolve_dtype(self):
        if self.method in _INT_METHODS:
            if self.dtype_backend == "pyarrow":
                return pd.ArrowDtype(pa.int64())
            return pd.Int64Dtype()
        elif self.method in _STR_METHODS:
            if self.dtype_backend == "pyarrow":
                return pd.ArrowDtype(pa.string())
            return np.dtype(object)
        else:
            return np.dtype(object)


def image_decode(
    series: "Series",
    errors: Literal["ignore", "raise"] = "ignore",
    dtype_backend: Optional[Literal["pyarrow"]] = None,
) -> "Series":
    if errors not in ("ignore", "raise"):
        raise ValueError(f"errors must be 'ignore' or 'raise', got {errors!r}")
    op = SeriesImageMethods(method="decode", errors=errors, dtype_backend=dtype_backend)
    return op(series)


def image_property(
    series: "Series",
    prop_name: str,
    dtype_backend: Optional[Literal["pyarrow"]] = None,
) -> "Series":
    op = SeriesImageMethods(method=prop_name, dtype_backend=dtype_backend)
    return op(series)
