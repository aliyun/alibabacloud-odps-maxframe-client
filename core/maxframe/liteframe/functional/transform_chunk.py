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

"""LiteFrame.transform_chunk() -- per-sub-chunk UDF (expression-based)."""

from maxframe.liteframe.core import drop_alignment_id
from maxframe.liteframe.functional.apply import groupby_transform_op
from maxframe.liteframe.functional.type_infer import infer_output_dtypes
from maxframe.liteframe.functional.utils import (
    build_udf_projections,
    pop_check_output_dtypes,
    validate_single_output,
)
from maxframe.liteframe.operators.project import LiteFrameProjection


def transform_chunk(
    self,
    func,
    *,
    input_frame_type="pandas",
    dtypes=None,
    skip_infer=False,
    args=(),
    placeback=True,
    **kwargs
):
    """Apply func per sub-chunk (execution fragment). Output must have same row count.

    Unlike ``apply_chunk``, the output row count must match the input
    (transform semantics).  ``func`` receives one *sub-chunk* — the execution
    fragment the engine hands it — and must return a DataFrame with the same
    number of rows.

    Sub-chunk boundaries are implementation-defined (a parquet row group, a
    file, ...) and finer under streaming, so ``func`` must be fragment-local:
    anything that observes all rows at once (``cum_sum``, ``rank``, head/tail
    slices) restarts inside every sub-chunk.  LiteFrame's own window functions
    (``cumsum``, ``rank``, ...) carry their boundary-handoff strategy; calling
    such logic inside ``func`` means accepting that the boundaries may not be
    the ones you want.  Use ``apply_chunk`` for explicit whole-chunk semantics.

    Parameters
    ----------
    func : callable or callable class
        Function taking one sub-chunk and returning a DataFrame with the same
        number of rows; invoked once per sub-chunk.  A callable class is
        instantiated and closed per sub-chunk and receives ``end=True``.
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
    placeback : bool, default True
        ``transform_chunk`` is a 1:1 per-sub-chunk projection and is always
        row-aligned with the input.  When True (default) the output
        inherits the input's ``alignment_id`` (usable for aligned column
        assignment / horizontal concat without a shuffle).  When False the
        output keeps a fresh alignment id (metadata-only; no layout change).

    Returns
    -------
    LiteFrame
        Lazy LiteFrame with the same row count as the input, where each
        value is the per-sub-chunk transform result.  Call ``.execute()`` to
        materialize.

    See Also
    --------
    transform : Transform per column or per row.
    apply_chunk : Apply a function per execution chunk.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})

    Scale every column within each sub-chunk by a constant (row count
    preserved):

    >>> result = lf.transform_chunk(lambda df: df * 2)
    >>> result.execute()
       a   b
    0  2  20
    1  4  40
    2  6  60

    Replace columns with a derived schema (still same row count):

    >>> result = lf.transform_chunk(
    ...     lambda df: df.assign(sum=df['a'] + df['b'])[['sum']],
    ...     dtypes={'sum': int})
    >>> result.execute()
       sum
    0   11
    1   22
    2   33
    """
    check_output_dtypes = pop_check_output_dtypes(kwargs)

    inferred = infer_output_dtypes(
        func,
        self.dtypes,
        dtypes=dtypes,
        skip_infer=skip_infer,
        args=args,
        kwargs=kwargs,
        input_frame_type=input_frame_type,
    )
    validate_single_output(inferred, "transform_chunk")
    out_dtypes = inferred[0] if inferred else self.dtypes

    projections = build_udf_projections(
        self,
        func,
        "transform_chunk",
        out_dtypes,
        input_frame_type=input_frame_type,
        check_output_dtypes=check_output_dtypes,
        args=args,
        kwargs=kwargs,
    )
    proj_op = LiteFrameProjection(projections=projections)
    result = proj_op(self)
    if not placeback:
        # transform_chunk is 1:1 chunk-preserving by construction; the flag
        # only controls alignment_id inheritance (no layout work to skip).
        drop_alignment_id(result)
    return result


def groupby_transform_chunk(
    self,
    func,
    *,
    dtypes=None,
    skip_infer=False,
    input_frame_type="pandas",
    args=(),
    placeback=True,
    **kwargs
):
    """GroupBy.transform_chunk() -- per-group-sub-chunk transform.

    Applies ``func`` to each group's execution chunk independently.  As with
    the ungrouped ``transform_chunk``, the output row count must match the
    input and ``func`` sees one sub-chunk of one group, so it must be
    fragment-local as well.

    Parameters
    ----------
    func : callable or callable class
        Function taking a per-group sub-chunk and returning a DataFrame with
        the same number of rows.  A callable class is instantiated and
        closed once per call.
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
    placeback : bool, default True
        When True (default) the per-group-chunk result is restored to the
        input row order and chunking and inherits the input's
        ``alignment_id``.  When False the output stays in reducer order
        with a fresh alignment id.

    Returns
    -------
    LiteFrame
        Lazy LiteFrame with the same row count as the input, where each
        value is the per-group transform result.  Call ``.execute()`` to
        materialize.

    See Also
    --------
    groupby_transform : Per-group column transform.
    groupby_apply_chunk : Per-group-chunk application.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'key': ['a', 'a', 'b', 'b'], 'val': [1, 2, 3, 4]})

    Scale each group's chunk by 2 (row count preserved):

    >>> result = lf.groupby('key').transform_chunk(lambda g: g * 2,
    ...                                            dtypes={'val': int})
    >>> result.execute()
       val
    0    2
    1    4
    2    6
    3    8
    """
    return groupby_transform_op(
        self,
        func,
        "transform_chunk",
        "groupby_transform_chunk",
        dtypes,
        skip_infer,
        input_frame_type,
        args,
        kwargs,
        placeback=placeback,
    )
