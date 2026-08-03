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


def _install():
    from maxframe.liteframe.core import LITEFRAME_TYPE
    from maxframe.liteframe.indexing.getitem import liteframe_getitem
    from maxframe.liteframe.indexing.iloc import (
        LiteFrameIloc,
        LiteFrameIlocGetItem,
        LiteFrameIlocSetItem,
    )
    from maxframe.liteframe.indexing.select import assign, drop, rename, select_dispatch
    from maxframe.liteframe.indexing.setitem import liteframe_setitem

    for cls in LITEFRAME_TYPE:
        setattr(cls, "__getitem__", liteframe_getitem)
        setattr(cls, "__setitem__", liteframe_setitem)
        setattr(cls, "select", select_dispatch)
        setattr(cls, "drop", drop)
        setattr(cls, "rename", rename)
        setattr(cls, "assign", assign)
        setattr(cls, "iloc", property(fget=lambda self: LiteFrameIloc(self)))


_install()
del _install
