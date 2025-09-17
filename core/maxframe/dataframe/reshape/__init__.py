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

from .melt import melt
from .pivot import pivot
from .pivot_table import pivot_table
from .stack import stack
from .unstack import unstack


def _install():
    from ..core import DATAFRAME_TYPE, SERIES_TYPE

    for t in DATAFRAME_TYPE:
        setattr(t, "melt", melt)
        setattr(t, "pivot", pivot)
        setattr(t, "pivot_table", pivot_table)
        setattr(t, "stack", stack)
        setattr(t, "unstack", unstack)

    for t in SERIES_TYPE:
        setattr(t, "unstack", unstack)


_install()
del _install
