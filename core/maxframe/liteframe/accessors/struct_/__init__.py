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

from maxframe.liteframe.accessors.struct_.core import (
    LiteFrameStructExpr,
    struct_method_to_handlers,
)

# Export for public use
methods = set(struct_method_to_handlers.keys())
del struct_method_to_handlers


def _install():
    """Install struct accessor on LiteFrame and LiteFrameData."""
    from maxframe.core import CachedAccessor
    from maxframe.liteframe.accessors.struct_.accessor import StructAccessor
    from maxframe.liteframe.accessors.struct_.core import struct_method_to_handlers
    from maxframe.liteframe.core import LITEFRAME_TYPE
    from maxframe.liteframe.initializer import LiteFrame as LiteFrameInit

    for method in struct_method_to_handlers:
        if not hasattr(StructAccessor, method):
            StructAccessor._register(method)

    ca = CachedAccessor("struct", StructAccessor)
    for liteframe in LITEFRAME_TYPE:
        liteframe.struct = ca
    LiteFrameInit.struct = ca


_install()
del _install
