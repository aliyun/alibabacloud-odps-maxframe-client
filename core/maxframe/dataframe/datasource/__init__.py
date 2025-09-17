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

from .core import PandasDataSourceOperator
from .from_dict import dataframe_from_dict
from .from_records import from_records
from .from_tensor import dataframe_from_tensor, series_from_tensor


def _install():
    from ..core import DATAFRAME_TYPE, SERIES_TYPE

    for t in DATAFRAME_TYPE:
        t.from_dict = staticmethod(dataframe_from_dict)
        t.from_records = staticmethod(from_records)
        t.from_tensor = staticmethod(dataframe_from_tensor)
    for t in SERIES_TYPE:
        t.from_tensor = staticmethod(series_from_tensor)


_install()
del _install
