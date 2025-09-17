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

from .core import SeriesDatetimeMethod, datetime_method_to_handlers

methods = set(datetime_method_to_handlers.keys())
del datetime_method_to_handlers


def _install():
    from ....core import CachedAccessor
    from ...core import SERIES_TYPE
    from .accessor import DatetimeAccessor
    from .core import datetime_method_to_handlers

    for method in datetime_method_to_handlers:
        if not hasattr(DatetimeAccessor, method):
            DatetimeAccessor._register(method)
    for series in SERIES_TYPE:
        series.dt = CachedAccessor("dt", DatetimeAccessor)


_install()
del _install
