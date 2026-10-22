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
    from maxframe.liteframe.window.cumulative import cummax, cummin, cumprod, cumsum
    from maxframe.liteframe.window.diff import diff
    from maxframe.liteframe.window.expanding import expanding
    from maxframe.liteframe.window.fill import bfill, ffill
    from maxframe.liteframe.window.pct_change import pct_change
    from maxframe.liteframe.window.rolling import rolling
    from maxframe.liteframe.window.shift import shift

    for cls in LITEFRAME_TYPE:
        setattr(cls, "shift", shift)
        setattr(cls, "diff", diff)
        setattr(cls, "pct_change", pct_change)
        setattr(cls, "rolling", rolling)
        setattr(cls, "cumsum", cumsum)
        setattr(cls, "cummax", cummax)
        setattr(cls, "cummin", cummin)
        setattr(cls, "cumprod", cumprod)
        setattr(cls, "expanding", expanding)
        setattr(cls, "ffill", ffill)
        setattr(cls, "bfill", bfill)


_install()
del _install
