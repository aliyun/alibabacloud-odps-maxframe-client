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

from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.functional.apply import apply, groupby_apply
from maxframe.liteframe.functional.apply_chunk import apply_chunk, groupby_apply_chunk
from maxframe.liteframe.functional.flat_apply import flat_apply, groupby_flat_apply
from maxframe.liteframe.functional.map import map_
from maxframe.liteframe.functional.transform import groupby_transform, transform
from maxframe.liteframe.functional.transform_chunk import (
    groupby_transform_chunk,
    transform_chunk,
)
from maxframe.liteframe.groupby.core import LiteFrameGroupBy


def _install():
    for cls in LITEFRAME_TYPE:
        setattr(cls, "map", map_)
        setattr(cls, "apply", apply)
        setattr(cls, "apply_chunk", apply_chunk)
        setattr(cls, "flat_apply", flat_apply)
        setattr(cls, "transform", transform)
        setattr(cls, "transform_chunk", transform_chunk)

    # GroupBy methods
    LiteFrameGroupBy.apply = groupby_apply
    LiteFrameGroupBy.flat_apply = groupby_flat_apply
    LiteFrameGroupBy.apply_chunk = groupby_apply_chunk
    LiteFrameGroupBy.transform = groupby_transform
    LiteFrameGroupBy.transform_chunk = groupby_transform_chunk


_install()
del _install
