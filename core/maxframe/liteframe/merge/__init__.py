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

from maxframe.liteframe.merge.merge import merge  # noqa: F401


def _install():
    from maxframe.liteframe.core import LITEFRAME_TYPE
    from maxframe.liteframe.merge.merge import merge

    for cls in LITEFRAME_TYPE:
        setattr(cls, "merge", merge)


_install()
del _install
