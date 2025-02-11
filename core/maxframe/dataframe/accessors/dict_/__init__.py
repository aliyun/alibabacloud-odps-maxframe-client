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
    from .accessor import DictAccessor
    from .contains import series_dict_contains
    from .getitem import series_dict_getitem, series_dict_getitem_with_key_error
    from .length import series_dict_length
    from .remove import series_dict_remove
    from .setitem import series_dict_setitem

    dict_method_to_handlers = {
        "__getitem__": series_dict_getitem_with_key_error,
        "get": series_dict_getitem,
        "__setitem__": series_dict_setitem,
        "len": series_dict_length,
        "remove": series_dict_remove,
        "contains": series_dict_contains,
    }

    for name, handler in dict_method_to_handlers.items():
        DictAccessor._register(name, handler)

    for series in SERIES_TYPE:
        series.dict = CachedAccessor("dict", DictAccessor)


_install()
del _install
