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

import functools


def _wrap_eq():
    from maxframe.liteframe.arithmetic.comparison import eq

    @functools.wraps(eq)
    def call(lf, other, **kw):
        from maxframe.core.mode import is_build_mode
        from maxframe.liteframe.core import LITEFRAME_TYPE

        if is_build_mode():
            return lf is other
        # Return NotImplemented for LiteFrame-vs-LiteFrame so that
        # Python falls back to identity comparison. This is critical
        # for WeakSet/WeakKeyDictionary operations (ref.__eq__) which
        # need a bool result, not a LiteFrame.
        # Use the .eq() method for element-wise LiteFrame comparison.
        if isinstance(other, LITEFRAME_TYPE):
            return NotImplemented
        return eq(lf, other, **kw)

    return call


def _wrap_comparison(func):
    @functools.wraps(func)
    def call(lf, other, **kw):
        from maxframe.liteframe.core import LITEFRAME_TYPE

        if isinstance(other, LITEFRAME_TYPE):
            return NotImplemented
        return func(lf, other, **kw)

    return call


def _install():
    from maxframe.liteframe.arithmetic.between import between
    from maxframe.liteframe.arithmetic.binary import (
        add,
        div,
        floordiv,
        lshift,
        mod,
        mul,
        pow,
        radd,
        rdiv,
        rfloordiv,
        rlshift,
        rmod,
        rmul,
        rpow,
        rrshift,
        rshift,
        rsub,
        sub,
    )
    from maxframe.liteframe.arithmetic.clip import clip
    from maxframe.liteframe.arithmetic.comparison import eq, ge, gt, le, lt, ne
    from maxframe.liteframe.arithmetic.logical import and_, invert, or_, xor
    from maxframe.liteframe.arithmetic.ufunc import __array_ufunc__
    from maxframe.liteframe.arithmetic.unary import abs_, neg, pos, round_
    from maxframe.liteframe.arithmetic.where import case_when, mask, where
    from maxframe.liteframe.core import LITEFRAME_TYPE, LiteFrame

    for cls in LITEFRAME_TYPE:
        # Binary arithmetic
        setattr(cls, "add", add)
        setattr(cls, "radd", radd)
        setattr(cls, "sub", sub)
        setattr(cls, "rsub", rsub)
        setattr(cls, "mul", mul)
        setattr(cls, "rmul", rmul)
        setattr(cls, "div", div)
        setattr(cls, "rdiv", rdiv)
        setattr(cls, "floordiv", floordiv)
        setattr(cls, "rfloordiv", rfloordiv)
        setattr(cls, "mod", mod)
        setattr(cls, "rmod", rmod)
        setattr(cls, "pow", pow)
        setattr(cls, "rpow", rpow)
        # Bitwise shift
        setattr(cls, "lshift", lshift)
        setattr(cls, "rlshift", rlshift)
        setattr(cls, "rshift", rshift)
        setattr(cls, "rrshift", rrshift)
        # Operator overloads
        setattr(cls, "__add__", add)
        setattr(cls, "__radd__", radd)
        setattr(cls, "__sub__", sub)
        setattr(cls, "__rsub__", rsub)
        setattr(cls, "__mul__", mul)
        setattr(cls, "__rmul__", rmul)
        setattr(cls, "__truediv__", div)
        setattr(cls, "__rtruediv__", rdiv)
        setattr(cls, "__floordiv__", floordiv)
        setattr(cls, "__rfloordiv__", rfloordiv)
        setattr(cls, "__mod__", mod)
        setattr(cls, "__rmod__", rmod)
        setattr(cls, "__pow__", pow)
        setattr(cls, "__rpow__", rpow)
        # Bitwise shift operators
        setattr(cls, "__lshift__", lshift)
        setattr(cls, "__rlshift__", rlshift)
        setattr(cls, "__rshift__", rshift)
        setattr(cls, "__rrshift__", rrshift)
        # Unary
        setattr(cls, "__neg__", neg)
        setattr(cls, "__pos__", pos)
        setattr(cls, "__abs__", abs_)
        setattr(cls, "round", round_)
        setattr(cls, "__array_ufunc__", __array_ufunc__)
        # Comparison methods
        setattr(cls, "eq", eq)
        setattr(cls, "ne", ne)
        setattr(cls, "lt", lt)
        setattr(cls, "le", le)
        setattr(cls, "gt", gt)
        setattr(cls, "ge", ge)
        # Logical
        setattr(cls, "__and__", and_)
        setattr(cls, "__or__", or_)
        setattr(cls, "__xor__", xor)
        setattr(cls, "__invert__", invert)
        # Elementwise transforms
        setattr(cls, "clip", clip)
        setattr(cls, "between", between)
        # Conditional replacement
        setattr(cls, "where", where)
        setattr(cls, "mask", mask)
        setattr(cls, "case_when", case_when)

    # Comparison dunder operators: only on LiteFrame, not LiteFrameData.
    # LiteFrameData is used internally by WeakSet and other collections
    # that rely on object identity via __eq__ returning bool.
    setattr(LiteFrame, "__eq__", _wrap_eq())
    setattr(LiteFrame, "__ne__", _wrap_comparison(ne))
    setattr(LiteFrame, "__lt__", _wrap_comparison(lt))
    setattr(LiteFrame, "__le__", _wrap_comparison(le))
    setattr(LiteFrame, "__gt__", _wrap_comparison(gt))
    setattr(LiteFrame, "__ge__", _wrap_comparison(ge))


_install()
del _install
