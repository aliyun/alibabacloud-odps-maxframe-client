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


from maxframe.liteframe.arithmetic.core import (
    _BOOL,
    _build_fused_projection,
    _resolve_column_expr,
)
from maxframe.liteframe.expressions import (
    LiteFrameAnd,
    LiteFrameGe,
    LiteFrameGt,
    LiteFrameLe,
    LiteFrameLiteral,
    LiteFrameLt,
    NamedExpr,
)

_INCLUSIVE_OPS = {
    "both": (LiteFrameGe, LiteFrameLe),
    "left": (LiteFrameGe, LiteFrameLt),
    "right": (LiteFrameGt, LiteFrameLe),
    "neither": (LiteFrameGt, LiteFrameLt),
}


def between(lf, left, right, inclusive="both"):
    """Return boolean LiteFrame indicating whether each value is between
    *left* and *right*.

    Parameters
    ----------
    left : scalar
        Left boundary.
    right : scalar
        Right boundary.
    inclusive : {"both", "left", "right", "neither"}, default "both"
        Include boundaries.

    Returns
    -------
    LiteFrame
        A boolean LiteFrame.

    See Also
    --------
    lt : Element-wise less-than comparison.
    gt : Element-wise greater-than comparison.
    le : Element-wise less-than-or-equal comparison.
    ge : Element-wise greater-than-or-equal comparison.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3, 4, 5]})
    >>> result = lf.between(2, 4)
    >>> result.execute()
           a
    0  False
    1   True
    2   True
    3   True
    4  False
    """
    ops = _INCLUSIVE_OPS.get(inclusive)
    if ops is None:
        raise ValueError(
            "Inclusive has to be either string of 'both',"
            "'left', 'right', or 'neither'."
        )
    left_op, right_op = ops

    projections = []
    for col_name in lf.columns:
        col_dtype = lf.dtypes[col_name]
        col_ref = _resolve_column_expr(lf, col_name, col_dtype)
        left_lit = LiteFrameLiteral(value=left, dtype=col_dtype)
        right_lit = LiteFrameLiteral(value=right, dtype=col_dtype)
        expr = LiteFrameAnd(
            left=left_op(left=col_ref, right=left_lit, dtype=_BOOL),
            right=right_op(left=col_ref, right=right_lit, dtype=_BOOL),
            dtype=_BOOL,
        )
        projections.append(NamedExpr(name=col_name, expr=expr))

    return _build_fused_projection(lf, projections)
