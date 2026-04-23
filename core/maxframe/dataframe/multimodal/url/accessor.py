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

from ....core import BaseMaxFrameAccessor
from ...core import SERIES_TYPE
from .core import url_download

if TYPE_CHECKING:
    from ...core import Series


class UrlAccessor(BaseMaxFrameAccessor):
    obj: "Series"
    _api_count: int = 1  # download

    def __init__(self, series: "Series"):
        if not isinstance(series, SERIES_TYPE):
            raise AttributeError("Can only use .url accessor with Series")
        super().__init__(series)

    def download(
        self,
        storage_options: Optional[Dict[str, Any]] = None,
        errors: Literal["ignore", "raise"] = "ignore",
        dtype_backend: Optional[Literal["pyarrow"]] = None,
    ) -> "Series":
        return url_download(
            self.obj,
            storage_options=storage_options,
            errors=errors,
            dtype_backend=dtype_backend,
        )
