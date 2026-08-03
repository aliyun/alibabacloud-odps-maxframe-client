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
    LiteFrameAdd,
    LiteFrameDiv,
    LiteFrameFloorDiv,
    LiteFrameMod,
    LiteFrameMul,
    LiteFramePow,
    LiteFrameSub,
)


def add(lf, other):
    return _build_binary_projection(lf, LiteFrameAdd, other)


def radd(lf, other):
    return _build_binary_projection(lf, LiteFrameAdd, other)


def sub(lf, other):
    return _build_binary_projection(lf, LiteFrameSub, other)


def rsub(lf, other):
    return _build_binary_projection(lf, LiteFrameSub, other, reverse=True)


def mul(lf, other):
    return _build_binary_projection(lf, LiteFrameMul, other)


def rmul(lf, other):
    return _build_binary_projection(lf, LiteFrameMul, other)


def div(lf, other):
    return _build_binary_projection(lf, LiteFrameDiv, other)


def rdiv(lf, other):
    return _build_binary_projection(lf, LiteFrameDiv, other, reverse=True)


def floordiv(lf, other):
    return _build_binary_projection(lf, LiteFrameFloorDiv, other)


def rfloordiv(lf, other):
    return _build_binary_projection(lf, LiteFrameFloorDiv, other, reverse=True)


def mod(lf, other):
    return _build_binary_projection(lf, LiteFrameMod, other)


def rmod(lf, other):
    return _build_binary_projection(lf, LiteFrameMod, other, reverse=True)


def pow(lf, other):
    return _build_binary_projection(lf, LiteFramePow, other)


def rpow(lf, other):
    return _build_binary_projection(lf, LiteFramePow, other, reverse=True)
