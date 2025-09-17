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


def _install():
    from ....core import CachedAccessor
    from ...core import SERIES_TYPE
    from .accessor import ListAccessor
    from .getitem import series_list_getitem, series_list_getitem_with_index_error
    from .length import series_list_length

    list_method_to_handlers = {
        "__getitem__": series_list_getitem_with_index_error,
        "get": series_list_getitem,
        "len": series_list_length,
    }

    for name, handler in list_method_to_handlers.items():
        ListAccessor._register(name, handler)

    for series in SERIES_TYPE:
        series.list = CachedAccessor("list", ListAccessor)


_install()
del _install
