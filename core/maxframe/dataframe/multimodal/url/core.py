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

from typing import TYPE_CHECKING, Any, Dict, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import pyarrow as pa

from .... import opcodes
from ....core import OutputType
from ....serialization.serializables import DictField, StringField
from ...operators import DataFrameOperator, DataFrameOperatorMixin

if TYPE_CHECKING:
    from ...core import Series


class SeriesUrlMethods(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.SERIES_URL_METHODS

    method = StringField("method", default=None)
    storage_options = DictField("storage_options", default=None)
    errors = StringField("errors", default="ignore")
    dtype_backend = StringField("dtype_backend", default=None)

    def __init__(self, output_types=None, **kw):
        output_types = output_types or [OutputType.series]
        kw["_output_types"] = kw.get("_output_types") or output_types
        super().__init__(**kw)

    def __call__(self, series: "Series") -> "Series":
        # Add input validation for URL strings
        if self.method == "download" and not (
            series.dtype == np.dtype(object) or series.dtype == np.dtype(str)
        ):
            raise TypeError(
                f"Input series for url.download must be string-like, but got {series.dtype}"
            )
        # Determine output dtype based on dtype_backend
        if self.dtype_backend == "pyarrow":
            dtype = pd.ArrowDtype(pa.binary())
        else:
            dtype = np.dtype(object)
        return self.new_series(
            [series],
            shape=series.shape,
            dtype=dtype,
            index_value=series.index_value,
            name=series.name,
        )


def url_download(
    series: "Series",
    storage_options: Optional[Dict[str, Any]] = None,
    errors: Literal["ignore", "raise"] = "ignore",
    dtype_backend: Optional[Literal["pyarrow"]] = None,
) -> "Series":
    if errors not in ("ignore", "raise"):
        raise ValueError(f"errors must be 'ignore' or 'raise', got {errors!r}")
    op = SeriesUrlMethods(
        method="download",
        storage_options=storage_options,
        errors=errors,
        dtype_backend=dtype_backend,
    )
    return op(series)
