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

from maxframe.liteframe.arithmetic.core import _build_binary_projection
from maxframe.liteframe.expressions import (
    LiteFrameEq,
    LiteFrameGe,
    LiteFrameGt,
    LiteFrameLe,
    LiteFrameLt,
    LiteFrameNe,
)


def eq(lf, other):
    return _build_binary_projection(lf, LiteFrameEq, other)


def ne(lf, other):
    return _build_binary_projection(lf, LiteFrameNe, other)


def lt(lf, other):
    return _build_binary_projection(lf, LiteFrameLt, other)


def le(lf, other):
    return _build_binary_projection(lf, LiteFrameLe, other)


def gt(lf, other):
    return _build_binary_projection(lf, LiteFrameGt, other)


def ge(lf, other):
    return _build_binary_projection(lf, LiteFrameGe, other)
