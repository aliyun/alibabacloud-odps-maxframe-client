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

from maxframe.liteframe.accessors.dict_.core import (
    LiteFrameDictExpr,
    dict_method_to_handlers,
)

# Export for public use
methods = set(dict_method_to_handlers.keys())
del dict_method_to_handlers


def _install():
    """Install dict accessor on LiteFrame and LiteFrameData."""
    from maxframe.core import CachedAccessor
    from maxframe.liteframe.accessors.dict_.accessor import DictAccessor
    from maxframe.liteframe.accessors.dict_.core import dict_method_to_handlers
    from maxframe.liteframe.core import LITEFRAME_TYPE
    from maxframe.liteframe.initializer import LiteFrame as LiteFrameInit

    for method in dict_method_to_handlers:
        if not hasattr(DictAccessor, method):
            DictAccessor._register(method)

    ca = CachedAccessor("dict", DictAccessor)
    for liteframe in LITEFRAME_TYPE:
        liteframe.dict = ca
    LiteFrameInit.dict = ca


_install()
del _install
