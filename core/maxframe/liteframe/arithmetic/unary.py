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

from numbers import Number

import numpy as np

from maxframe.liteframe.arithmetic.core import _build_unary_projection
from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.expressions import LiteFrameAbs, LiteFrameNeg, LiteFrameRound


def neg(lf):
    return _build_unary_projection(lf, LiteFrameNeg)


def pos(lf):
    return lf


def abs_(lf):
    return _build_unary_projection(lf, LiteFrameAbs)


def round_(lf, decimals=0):
    return _build_unary_projection(lf, LiteFrameRound, decimals=decimals)


_UFUNC_MAP = {
    "absolute": abs_,
    "fabs": abs_,
    "negative": neg,
    "positive": pos,
    "round": round_,
    "around": round_,
}


def __array_ufunc__(lf, ufunc, method, *inputs, **kwargs):
    """NumPy ufunc dispatch for LiteFrame."""
    if method != "__call__":
        return NotImplemented

    for x in inputs:
        if not isinstance(x, (Number,)) and not isinstance(x, LITEFRAME_TYPE):
            try:
                np.asarray(x)
            except Exception:
                return NotImplemented

    handler = _UFUNC_MAP.get(ufunc.__name__)
    if handler is None:
        return NotImplemented

    # Find the LiteFrame input
    lf_input = None
    for x in inputs:
        if isinstance(x, LITEFRAME_TYPE):
            lf_input = x
            break

    if lf_input is None:
        return NotImplemented

    return handler(lf_input, **kwargs)
