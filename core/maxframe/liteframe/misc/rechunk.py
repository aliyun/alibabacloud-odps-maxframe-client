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

import operator

from maxframe.liteframe.operators.misc.rechunk import LiteFrameRechunk
from maxframe.tensor.rechunk.rechunk import chunk_size_type
from maxframe.typing_ import TileableType


def validate_axis(axis, tileable=None):
    if axis == "index":
        axis = 0
    elif axis == "columns":
        axis = 1

    illegal = False
    try:
        axis = operator.index(axis)
        if axis < 0 or (tileable is not None and axis >= tileable.ndim):
            illegal = True
    except TypeError:
        illegal = True

    if illegal:
        raise ValueError(f"No axis named {axis} for object type {type(tileable)}")
    return axis


def _assert_row_axis(axis):
    """LiteFrame is row-axis-only; reject any non-row axis at the API
    boundary so a column-axis rechunk/rebalance does not silently no-op."""
    if axis not in (None, 0):
        raise ValueError(
            f"LiteFrame only supports rechunking along axis 0, got axis={axis}"
        )


def rechunk(
    a: TileableType, chunk_size: chunk_size_type, reassign_worker=False, axis=None
):
    """
    Rechunk LiteFrame data.

    This function is used to reorganize the chunk structure of data, which can
    change how data is distributed in distributed computing. By adjusting chunk
    sizes, memory usage and computational performance can be optimized. LiteFrame
    is only ever chunked along the row axis (axis 0); the column axis always
    stays a single chunk. Note that this function is only effective in DPE for
    now.

    Parameters
    ----------
    chunk_size : chunk_size_type
        New chunk size configuration. Can be one of the following forms:

        - Integer: row chunk size, e.g. ``100`` rechunks rows into chunks of 100.
        - Tuple: per-dimension sizes; only the first element (rows) takes
          effect. For instance, ``(100, 2)`` rechunks rows into chunks of 100.
        - Dictionary: chunk sizes for specific dimensions, for instance,
          ``{0: 100}`` rechunks dimension 0 into 100 rows per chunk, and
          ``{0: (100, 100, 100)}`` rechunks dimension 0 into 3 chunks where
          each chunk has 100 rows.

    reassign_worker : bool, default False
        Reserved for future use.
    axis : int, optional
        If specified, ``chunk_size`` applies only to that axis. Internally
        normalized to the dict form ``{axis: chunk_size}``. Cannot be used
        together with a dict ``chunk_size``.

    Returns
    -------
    TileableType
        The rechunked LiteFrame object.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.read_odps_table("test_table")
    >>> # Rechunk to 100 rows per chunk
    >>> lf_rechunked = lf.rechunk(100)
    >>> # Rechunk only axis 0 (rows)
    >>> lf_rechunked = lf.rechunk(100, axis=0)
    """
    if axis is not None:
        axis = validate_axis(axis, a)
    _assert_row_axis(axis)
    if isinstance(chunk_size, dict):
        bad_keys = [k for k in chunk_size if k != 0]
        if bad_keys:
            raise ValueError(
                "LiteFrame only supports rechunking along axis 0, "
                f"got dict keys {bad_keys}"
            )
    chunk_size, _ = LiteFrameRechunk.normalize_rechunk_axis(
        axis, chunk_size, validate_axis, axis, a
    )
    op = LiteFrameRechunk(chunk_size=chunk_size, reassign_worker=reassign_worker)
    return op(a)


def rebalance(lf: TileableType, axis=0, factor=None, num_partitions=None, rough=False):
    """
    Make data more balanced across entire cluster.

    LiteFrame is only ever chunked along the row axis (axis 0); ``axis`` is
    accepted for API parity with ``DataFrame.rebalance`` but should remain 0.

    Parameters
    ----------
    axis : int
        The axis to rebalance.
    factor : float
        Specified so the number of chunks after balance along the selected
        axis per plane is ``chunk_shape[axis] * factor``.
    num_partitions : int
        Specified so the number of chunks are at most
        num_partitions.
    rough : bool, default False
        Hint to rebalance without first resolving unknown row shapes.

        When True, rebalance may partition the row axis using a per-chunk
        size estimate (``rough_size_ratio``) instead of waiting for real
        (resolved) shapes. This only has an effect when the row shape is
        unknown **and** the source chunks carry such an estimate; currently
        that estimate is produced only by SIZE-mode ``read_odps_table``
        reads. In every other case -- known shapes, or an unknown row axis
        with no estimate -- ``rough=True`` silently falls back to the normal
        path (resolving real shapes then rechunking). The estimate is
        approximate, so the output is balanced by estimated share, not
        exactly equal in size.

    Returns
    -------
    LiteFrame
        Result of LiteFrame after rebalanced.
    """
    axis = validate_axis(axis, lf)
    _assert_row_axis(axis)

    LiteFrameRechunk.validate_rebalance_args(factor, num_partitions)

    op = LiteFrameRechunk(
        chunk_size=None,
        axis=axis,
        factor=factor,
        num_partitions=num_partitions,
        rough=rough,
    )
    return op(lf)
