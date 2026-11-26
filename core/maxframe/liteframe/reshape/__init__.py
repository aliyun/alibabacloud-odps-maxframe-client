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

from maxframe.liteframe.reshape import melt, pivot_table  # noqa: F401


def _install():
    from maxframe.liteframe.core import LITEFRAME_TYPE
    from maxframe.liteframe.reshape.melt import melt
    from maxframe.liteframe.reshape.pivot_table import pivot, pivot_table

    for cls in LITEFRAME_TYPE:
        setattr(cls, "pivot", pivot)
        setattr(cls, "pivot_table", pivot_table)
        setattr(cls, "melt", melt)


_install()
del _install
