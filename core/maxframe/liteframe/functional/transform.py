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

"""
LiteFrame.transform() -- axis=0 (operator-based, default)
or axis=1 (expression-based).
"""

from maxframe.liteframe.functional.apply import groupby_transform_op
from maxframe.liteframe.functional.type_infer import infer_output_dtypes
from maxframe.liteframe.functional.utils import (
    build_udf_projections,
    create_functional_op,
    merge_hidden_into_physical,
    pop_check_output_dtypes,
    propagate_hidden_metadata,
    validate_single_output,
    wrap_inference_func,
)
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.tensor.utils import validate_axis


def transform(
    self,
    func,
    axis=0,
    *,
    input_frame_type="pandas",
    dtypes=None,
    skip_infer=False,
    args=(),
    **kwargs
):
    """Transform per column (axis=0, default) or per row (axis=1).

    - ``axis=0`` (default): operator-based, per-column.  Columns are
      concatenated and ``func`` is applied per column; the output row count
      must match the input (transform semantics).  Uses
      ``LiteFrameFunctional``.  Mirrors ``pandas.DataFrame.transform`` which
      defaults to ``axis=0``.
    - ``axis=1``: expression-based, per-row.  The function receives each
      row and may return a dict or Series with any declared schema; use
      ``dtypes`` to name the output columns.

    Parameters
    ----------
    func : callable or callable class
        Function to apply.  For ``axis=0`` it receives a column (Series)
        and must return a transformed column of the same length.  For
        ``axis=1`` it receives a single row and may return a dict or
        Series declaring any output schema via ``dtypes``.

        If ``func`` is a class, it is instantiated once per chunk,
        ``__call__`` is invoked per column (``axis=0``) or per row
        (``axis=1``), and ``close()`` (if defined) is always called at
        the end of the chunk — even when ``__call__`` receives single
        rows/columns.
    axis : {0, 1}, default 0
        Axis along which to transform.  ``0`` operates per column, ``1``
        per row.
    input_frame_type : {"pandas", "polars"}, default "pandas"
        Frame type passed to ``func``.
    dtypes : dict or pd.Series, optional
        Expected output dtypes.
    skip_infer : bool, default False
        If True, skip dtype inference.
    args : tuple
        Extra positional arguments forwarded to ``func``.
    **kwargs
        Extra keyword arguments forwarded to ``func``.

    Returns
    -------
    LiteFrame
        Lazy LiteFrame with the same row count as the input (``axis=0``)
        or a schema declared via ``dtypes`` (``axis=1``).  Call
        ``.execute()`` to materialize.

    See Also
    --------
    apply : Apply a function per column or per row.
    transform_chunk : Transform per execution chunk.
    map : Apply a function element-wise.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})

    Transform per column (``axis=0``, default) — ``func`` receives each
    column as a Series and must return a same-length Series:

    >>> result = lf.transform(lambda col: col * 2)
    >>> result.execute()
       a   b
    0  2  20
    1  4  40
    2  6  60

    Transform per row (``axis=1``) — ``func`` receives each row as a
    Series and may return a dict with any declared schema.  Declare the
    output schema with ``dtypes``:

    >>> result = lf.transform(lambda row: {'ab': row['a'] + row['b']},
    ...                       axis=1, dtypes={'ab': int})
    >>> result.execute()
       ab
    0  11
    1  22
    2  33
    """
    axis = validate_axis(self.ndim, axis, argname="axis")
    if axis == 1:
        # Expression-based: per-row transform
        check_output_dtypes = pop_check_output_dtypes(kwargs)
        # Wrap the user func so mock type inference honours the per-row contract:
        # transform(axis=1) passes a single row to the UDF, not the whole frame.
        # infer_output_dtypes stays generic (it always calls func(mock_df)); the
        # wrapper owns how the UDF is actually invoked. Without it a row UDF such
        # as ``lambda row: {"x": row["a"] + row["b"]}`` is inferred from a full
        # mock DataFrame and can silently produce an empty output schema.
        infer_func = wrap_inference_func(
            func, "transform_row", input_frame_type, args, kwargs
        )
        inferred = infer_output_dtypes(
            infer_func,
            self.dtypes,
            dtypes=dtypes,
            skip_infer=skip_infer,
            args=(),
            kwargs={},
            input_frame_type=input_frame_type,
        )
        validate_single_output(inferred, "transform(axis=1)")
        out_dtypes = inferred[0] if inferred else self.dtypes

        projections = build_udf_projections(
            self,
            func,
            "transform_row",
            out_dtypes,
            input_frame_type=input_frame_type,
            check_output_dtypes=check_output_dtypes,
            args=args,
            kwargs=kwargs,
        )
        proj_op = LiteFrameProjection(projections=projections)
        return proj_op(self)
    else:
        # Operator-based: per-column transform (needs block concat)
        op, inferred = create_functional_op(
            func,
            mode="transform",
            input_dtypes=self.dtypes,
            dtypes=dtypes,
            skip_infer=skip_infer,
            input_frame_type=input_frame_type,
            args=args,
            kwargs=kwargs,
            axis=0,
        )
        validate_single_output(inferred, "transform(axis=0)")
        out_dtypes = inferred[0] if inferred else self.dtypes
        out_meta = propagate_hidden_metadata(self)
        physical_dtypes, out_meta, n_cols = merge_hidden_into_physical(
            self, out_dtypes, out_meta
        )
        return op.new_liteframe(
            [self],
            shape=(self.shape[0], n_cols),
            physical_dtypes=physical_dtypes,
            frame_metadata=out_meta,
        )


def groupby_transform(
    self,
    func,
    *,
    dtypes=None,
    skip_infer=False,
    input_frame_type="pandas",
    args=(),
    **kwargs
):
    """GroupBy.transform() -- per-group column transform.

    Applies ``func`` to each group independently and returns a LiteFrame
    with the same row count as the input (transform semantics).  The output
    row order follows reducer/hash-partition order and is not aligned with
    the input frame.

    Parameters
    ----------
    func : callable or callable class
        Function receiving a per-group column (Series) and returning a
        transformed column of the same length.  ``func`` is called once
        per (group, column) pair with a single-column Series, not a
        whole-group DataFrame.  Use ``lambda g: g - g.mean()``, not
        ``g['col'] - g['col'].mean()``.

        If ``func`` is a class, it is instantiated once per group,
        ``__call__`` is invoked per (group, column), and ``close()``
        (if defined) is always called at the end of each group.
    dtypes : dict or pd.Series, optional
        Expected output dtypes.
    skip_infer : bool, default False
        If True, skip dtype inference.
    input_frame_type : {"pandas", "polars"}, default "pandas"
        Frame type passed to ``func``.
    args : tuple
        Extra positional arguments forwarded to ``func``.
    **kwargs
        Extra keyword arguments forwarded to ``func``.

    Returns
    -------
    LiteFrame
        Lazy LiteFrame with the same row count as the input, where each
        value is the per-group transform result.  Call ``.execute()`` to
        materialize.

    See Also
    --------
    groupby_apply : Apply a function to each group as a whole frame.
    groupby_transform_chunk : Per-group-chunk transform.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'key': ['a', 'a', 'b', 'b'], 'val': [1, 2, 3, 4]})

    Center each value by its group mean (row count preserved):

    >>> result = lf.groupby('key').transform(lambda g: g - g.mean(),
    ...                                      dtypes={'val': float})
    >>> result.execute()
       val
    0 -0.5
    1  0.5
    2 -0.5
    3  0.5
    """
    return groupby_transform_op(
        self,
        func,
        "transform",
        "groupby_transform",
        dtypes,
        skip_infer,
        input_frame_type,
        args,
        kwargs,
    )
