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

from maxframe.liteframe.accessors.datetime_.core import (
    LiteFrameDatetimeExpr,
    datetime_method_to_handlers,
    datetime_multi_method_to_handlers,
)

# Export for public use
methods = set(datetime_method_to_handlers.keys())
del datetime_method_to_handlers


def _install():
    """Install dt accessor on LiteFrame and LiteFrameData."""
    from maxframe.core import CachedAccessor
    from maxframe.liteframe.accessors.datetime_.accessor import DatetimeAccessor
    from maxframe.liteframe.accessors.datetime_.core import (
        datetime_method_to_handlers,
        datetime_multi_method_to_handlers,
    )
    from maxframe.liteframe.core import LITEFRAME_TYPE
    from maxframe.liteframe.initializer import LiteFrame as LiteFrameInit

    # Register all single-column handler methods to accessor class
    for method in datetime_method_to_handlers:
        if not hasattr(DatetimeAccessor, method):
            DatetimeAccessor._register(method)

    # Register all multi-column handler methods to accessor class
    for method in datetime_multi_method_to_handlers:
        if not hasattr(DatetimeAccessor, method):
            DatetimeAccessor._register_multi(method)

    # Attach .dt accessor to core LiteFrame/LiteFrameData and the initializer
    ca = CachedAccessor("dt", DatetimeAccessor)
    for liteframe in LITEFRAME_TYPE:
        liteframe.dt = ca
    LiteFrameInit.dt = ca


_install()
del _install
