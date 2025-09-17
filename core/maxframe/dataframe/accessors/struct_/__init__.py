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
    from .accessor import StructAccessor
    from .dtypes import struct_dtypes
    from .field import struct_field

    struct_method_to_handlers = {
        "dtypes": struct_dtypes,
        "field": struct_field,
    }
    property_names = {"dtypes"}

    for name, handler in struct_method_to_handlers.items():
        StructAccessor._register(name, handler, is_property=name in property_names)

    for series in SERIES_TYPE:
        series.struct = CachedAccessor("struct", StructAccessor)


_install()
del _install
