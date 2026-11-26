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

"""LiteFrame.map() — element-wise UDF application."""

import logging

import pandas as pd

from maxframe.liteframe.expressions import LiteFrameColumn, LiteFrameUDFExpr, NamedExpr
from maxframe.liteframe.functional.type_infer import (
    _get_pa_type,
    _normalize_to_arrow_dtype,
    build_mock_value,
    infer_scalar_dtype_from_annotation,
)
from maxframe.liteframe.functional.utils import (
    _build_scheduling_hint_from_func,
    call_user_func,
)
from maxframe.liteframe.operators.project import (
    LiteFrameProjection,
    _append_hidden_projections,
)

logger = logging.getLogger(__name__)


def _infer_map_col_dtypes(liteframe, func, args, kwargs):
    """Infer map output dtypes by calling func on one mock value per column."""
    col_dtypes = {}
    for col_name, col_dtype in liteframe.dtypes.items():
        try:
            pa_type = _get_pa_type(col_dtype)
            mock_val = build_mock_value(pa_type)

            result_val = call_user_func(func, mock_val, args, kwargs)
            if result_val is None:
                col_dtypes[col_name] = col_dtype
            elif isinstance(result_val, str):
                # pd.Series on a string yields a non-arrow StringDtype;
                # use pa.string() directly for arrow consistency.
                col_dtypes[col_name] = _normalize_to_arrow_dtype(str)
            else:
                col_dtypes[col_name] = _normalize_to_arrow_dtype(
                    pd.Series([result_val]).dtype
                )
        except Exception as exc:
            logger.debug(
                "map() dtype inference failed for column %r with %s: %s; "
                "falling back to input dtype %s",
                col_name,
                type(exc).__name__,
                exc,
                col_dtype,
            )
            col_dtypes[col_name] = col_dtype
    return col_dtypes


def map_(
    self, func, *, na_action=None, dtype=None, skip_infer=False, args=(), **kwargs
):
    """Apply func element-wise to every value in the LiteFrame.

    Each cell is transformed independently by ``func``; the output has the
    same shape and column names as the input.  Output dtypes come from
    ``dtype``, a scalar return annotation such as ``-> int``, or mock
    inference on a sample value per column.

    Parameters
    ----------
    func : callable or callable class
        Function to apply to each element. A scalar return annotation is
        used as the output dtype when ``dtype`` is omitted.

        If ``func`` is a class, it is instantiated once per chunk,
        ``__call__`` is invoked per element, and ``close()`` (if
        defined) is always called at the end of the chunk.
    na_action : str, optional
        If 'ignore', null values are propagated without calling func.
    dtype : dtype, optional
        Output dtype applied to all columns. Overrides a return annotation.
    skip_infer : bool
        If True, skip mock inference when neither dtype nor a return
        annotation is provided.
    args : tuple
        Positional arguments forwarded to func.
    **kwargs
        Keyword arguments. '_check_output_dtype' is popped for output
        validation; all others are forwarded to func.

    Returns
    -------
    LiteFrame
        Lazy LiteFrame with the same shape as the input, where each value
        is the result of applying ``func`` to the corresponding input cell.
        Call ``.execute()`` to materialize.

    See Also
    --------
    apply : Apply a function per column or per row.
    transform : Transform per column or per row.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
    >>> result = lf.map(lambda x: x * 2)
    >>> result.execute()
       a   b
    0  2  20
    1  4  40
    2  6  60

    Specify an explicit output dtype with ``dtype``:

    >>> result = lf.map(lambda x: float(x), dtype='float64')
    >>> result.execute()
         a     b
    0  1.0  10.0
    1  2.0  20.0
    2  3.0  30.0
    """
    if na_action is not None:
        raise NotImplementedError("na_action is not yet supported for LiteFrame.map()")

    check_output_dtype = kwargs.pop("_check_output_dtype", "raises")

    # Infer per-column output dtype
    if dtype is not None:
        resolved_dtype = _normalize_to_arrow_dtype(dtype)
        col_dtypes = {col: resolved_dtype for col in self.dtypes.index}
    else:
        annotated_dtype = infer_scalar_dtype_from_annotation(func)
        if annotated_dtype is not None:
            col_dtypes = {col: annotated_dtype for col in self.dtypes.index}
        elif skip_infer:
            col_dtypes = dict(self.dtypes)
        else:
            col_dtypes = _infer_map_col_dtypes(self, func, args, kwargs)

    # Build projections
    projections = []
    for col_name in self.dtypes.index:
        out_dtype = col_dtypes[col_name]
        udf_expr = LiteFrameUDFExpr(
            func=func,
            mode="map",
            operands=[LiteFrameColumn(name=col_name, dtype=self.dtypes[col_name])],
            output_names=[col_name],
            dtypes={col_name: out_dtype},
            input_frame_type=None,
            check_output_dtypes=check_output_dtype,
            args=args,
            kwargs=kwargs,
            scheduling_hint=_build_scheduling_hint_from_func(func),
        )
        projections.append(NamedExpr(name=col_name, expr=udf_expr, dtype=out_dtype))

    _append_hidden_projections(self, projections)

    proj_op = LiteFrameProjection(projections=projections)
    return proj_op(self)
