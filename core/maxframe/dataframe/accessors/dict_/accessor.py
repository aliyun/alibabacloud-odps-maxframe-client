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

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from ....core import BaseMaxFrameAccessor
from ....utils import ARROW_DTYPE_NOT_SUPPORTED

if TYPE_CHECKING:
    from ...core import Series


class DictAccessor(BaseMaxFrameAccessor):
    obj: "Series"
    _api_count: int = 0

    def __init__(self, series):
        super().__init__(series)
        if ARROW_DTYPE_NOT_SUPPORTED:
            raise ImportError("pd.ArrowDtype is not supported in current environment")

        if not isinstance(series.dtype, pd.ArrowDtype) or not isinstance(
            series.dtype.pyarrow_dtype, pa.MapType
        ):
            raise AttributeError("Can only use .dict accessor with dict values")
