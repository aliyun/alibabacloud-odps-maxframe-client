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

"""LiteFrame.apply_chunk() -- per-chunk UDF application (operator-based)."""

import warnings

from maxframe.liteframe.functional.apply import groupby_apply_op
from maxframe.liteframe.functional.utils import (
    build_functional_output,
    create_functional_op,
)


def _pop_batch_rows_warning(kwargs):
    """Pop _batch_rows from kwargs and warn if present."""
    _batch_rows = kwargs.pop("_batch_rows", None)
    if _batch_rows is not None:
        warnings.warn(
            "_batch_rows is not supported by the DPE backend and has no "
            "effect under that case. It is reserved for MCSQL backend.",
            stacklevel=3,
        )
    return _batch_rows


def apply_chunk(
    self,
    func,
    *,
    dtypes=None,
    skip_infer=False,
    input_frame_type="pandas",
    args=(),
    **kwargs,
):
    """Apply func per execution chunk. Row count may change.

    Unlike ``transform`` / ``transform_chunk``, the row count of the output
    is allowed to differ from the input — the function is applied once per
    execution chunk and the results are concatenated.  Hidden columns are
    dropped from the input before invocation; select them first if needed.

    Parameters
    ----------
    func : callable or callable class
        Function taking a DataFrame chunk and returning a DataFrame (or a
        tuple of DataFrames for multiple outputs).

        If ``func`` is a class, it is instantiated once per chunk,
        ``__call__(batch, end=True)`` is invoked once with ``end=True``
        (each call is a complete invocation that must flush its output),
        and ``close()`` (if defined) is always called at the end of the
        chunk.
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
        Lazy LiteFrame whose materialized output is the per-chunk
        concatenation of ``func`` results.  The row count may differ from
        the input.  Call ``.execute()`` to materialize.

    See Also
    --------
    apply : Apply a function per column or per row.
    transform_chunk : Transform per execution chunk.
    flat_apply : Apply a function per row, expanding to 0-N rows.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})

    Filter rows within each chunk (row count may change):

    >>> result = lf.apply_chunk(lambda df: df[df['a'] > 2],
    ...                         dtypes={'a': int, 'b': int})
    >>> result.execute()
       a   b
    0  3  30
    1  4  40

    Produce multiple outputs by returning a tuple of DataFrames:

    >>> results = lf.apply_chunk(lambda df: (df, df * 2),
    ...                          dtypes=[{'a': int, 'b': int},
    ...                                  {'a': int, 'b': int}])
    >>> len(results)
    2
    >>> results[0].execute()
       a   b
    0  1  10
    1  2  20
    2  3  30
    3  4  40
    """
    _pop_batch_rows_warning(kwargs)
    # Drop policy: hidden columns are not exposed to the per-chunk UDF and are
    # not carried into the output.  Drop them from the input at the API (where
    # the tileable carries ``frame_metadata``) so the executor sees visible
    # columns only; users who need a hidden column must ``select`` it first.
    hidden = sorted(self._hidden_columns)
    inp = self.drop(hidden) if hidden else self
    op, inferred = create_functional_op(
        func,
        mode="apply_chunk",
        input_dtypes=self.dtypes,
        dtypes=dtypes,
        skip_infer=skip_infer,
        input_frame_type=input_frame_type,
        args=args,
        kwargs=kwargs,
    )
    return build_functional_output(op, [inp], inferred)


def groupby_apply_chunk(
    self,
    func,
    *,
    dtypes=None,
    skip_infer=False,
    input_frame_type="pandas",
    args=(),
    **kwargs,
):
    """GroupBy.apply_chunk() -- per-group-chunk application.

    Applies ``func`` to each group's execution chunk independently.  As with
    the ungrouped ``apply_chunk``, the row count and row order may change.
    Hidden columns are not propagated (see ``groupby_apply`` for details).

    Parameters
    ----------
    func : callable or callable class
        Function taking a per-group DataFrame chunk and returning a
        DataFrame (or a tuple of DataFrames for multiple outputs).

        If ``func`` is a class, it is instantiated once per group,
        ``__call__(batch, end=True)`` is invoked once with ``end=True``
        (each call is a complete invocation that must flush its output),
        and ``close()`` (if defined) is always called at the end of each
        group.
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
        Lazy LiteFrame whose materialized output is the per-group
        concatenation of ``func`` results.  The row count may differ from
        the input.  Call ``.execute()`` to materialize.

    See Also
    --------
    groupby_apply : Apply a function to each group as a whole frame.
    groupby_transform_chunk : Per-group-chunk transform.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'key': ['a', 'a', 'b'], 'val': [1, 2, 3]})

    Filter rows within each group's chunk (row count may change):

    >>> result = lf.groupby('key')[['key', 'val']].apply_chunk(
    ...     lambda g: g[g['val'] > 1], dtypes={'key': str, 'val': int})
    >>> result.execute()
      key  val
    0   a    2
    1   b    3
    """
    _pop_batch_rows_warning(kwargs)
    return groupby_apply_op(
        self,
        func,
        "apply_chunk",
        dtypes,
        skip_infer,
        input_frame_type,
        args,
        kwargs,
    )
