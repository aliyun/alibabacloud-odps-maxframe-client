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

from typing import Tuple, Union

from maxframe import opcodes
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    Float64Field,
    Int64Field,
)
from maxframe.tensor.core import Tensor
from maxframe.tensor.operators import TensorOperator, TensorOperatorMixin
from maxframe.tensor.utils import validate_axis

chunk_size_type = Union[int, Tuple[int], Tuple[Tuple[int], ...]]


class TensorRechunk(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.RECHUNK

    chunk_size = AnyField("chunk_size", default=None)
    axis = Int64Field("axis", default=None)
    factor = Float64Field("factor", default=None)
    num_partitions = Int64Field("num_partitions", default=None)
    rough = BoolField("rough", default=False)

    def __call__(self, tensor: Tensor):
        return self.new_tensor([tensor], tensor.shape, order=tensor.order)

    @staticmethod
    def validate_rebalance_args(factor, num_partitions):
        """Validate that exactly one of ``factor`` / ``num_partitions`` is set."""
        if num_partitions is None and factor is None:
            raise ValueError("Need to specify num_partitions or factor")
        if num_partitions is not None and factor is not None:
            raise ValueError(
                "num_partitions and factor cannot be specified at the same time"
            )

    @staticmethod
    def normalize_rechunk_axis(axis, chunk_size, validate_axis, *validate_args):
        """Normalize an ``axis`` arg into a dict ``chunk_size``.

        Returns ``(chunk_size, axis)`` where ``chunk_size`` is a dict if ``axis``
        was given, and ``axis`` is always ``None`` (erased — the axis info lives
        only in the dict ``chunk_size``).
        """
        if axis is not None:
            if isinstance(chunk_size, dict):
                raise ValueError("Cannot specify axis when chunk_size is a dict")
            axis = validate_axis(*validate_args)
            chunk_size = {axis: chunk_size}
        return chunk_size, None


def rechunk(
    tensor: Tensor, chunk_size: chunk_size_type, reassign_worker=False, axis=None
) -> Tensor:
    """
    Rechunk tensor data.

    This function is used to reorganize the chunk structure of data, which can
    change how data is distributed in distributed computing. By adjusting chunk
    sizes, memory usage and computational performance can be optimized. Note that
    this function is only effective in DPE for now.

    Parameters
    ----------
    chunk_size : chunk_size_type
        New chunk size configuration. Can be one of the following forms:
        - Integer: Same chunk size for all dimensions
        - Tuple: Different chunk sizes for each dimension. For instance, (100, 100)
          rechunks dimension 0 and 1 into chunks with size 100 respectively, and
          ((100, 100, 100), (100,)) rechunks dimension 0 into 3 chunks with size
          100 for each chunk, and rechunks dimension 1 into one single chunk with
          size 100.
        - Dictionary: Chunk sizes for specific dimensions, for instance,
          `{0: 100}` rechunks dimension 0 into 100 elements per chunk, and
          `{0: (100, 100, 100)}` rechunks dimension 0 into 3 chunks where each
          chunk has size 100.
    reassign_worker : bool, default False
        Reserved for future use.
    axis : int, optional
        If specified, ``chunk_size`` applies only to that axis. Internally
        normalized to the dict form ``{axis: chunk_size}``. Cannot be used
        together with a dict ``chunk_size``.

    Returns
    -------
    TileableType
        The rechunked tensor object.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> t = mt.random.rand(1000, 1000)
    >>> # Rechunk to 100 rows and columns per chunk
    >>> t_rechunked = t.rechunk(100)
    >>> # Specify different chunk sizes for different dimensions
    >>> t_rechunked = t.rechunk((50, 60))
    >>> # Rechunk only axis 0
    >>> t_rechunked = t.rechunk(100, axis=0)
    """
    chunk_size, _ = TensorRechunk.normalize_rechunk_axis(
        axis, chunk_size, validate_axis, tensor.ndim, axis
    )
    op = TensorRechunk(
        chunk_size=chunk_size,
        reassign_worker=reassign_worker,
        dtype=tensor.dtype,
        sparse=tensor.issparse(),
    )
    return op(tensor)


def rebalance(tensor, axis=0, factor=None, num_partitions=None, rough=False):
    """
    Make data more balanced across entire cluster.

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
        Hint to rebalance without first resolving unknown axis shapes.

        When True, rebalance may partition the pivoting axis using a per-chunk
        size estimate instead of waiting for real (resolved) shapes. This only
        has an effect when the pivoting axis shape is unknown **and** the source
        chunks carry such an estimate; currently that estimate is produced only
        by SIZE-mode ``read_odps_table`` reads along the row axis (axis 0). The
        rough path is therefore restricted to ``axis=0`` -- on any other axis
        ``rough=True`` falls back to the normal path even when shapes are
        unknown, because the row-share estimate would collapse to an
        uninformative equal split. In every other case -- known shapes, or an
        unknown row axis with no estimate -- ``rough=True`` silently falls back
        to the normal path (resolving real shapes then rechunking). The estimate
        is approximate, so the output is balanced by estimated share, not
        exactly equal in size.

        .. note:: ``rough`` is infrastructure for the upcoming SIZE-mode
           ``read_odps_table`` -> rough rebalance pipeline. No production call
           site in the current corpus passes ``rough=True``; it is exercised by
           the producer + downstream-engagement tests until a consumer lands.

    Returns
    -------
    Tensor
        Result of tensor after rebalanced.
    """
    TensorRechunk.validate_rebalance_args(factor, num_partitions)

    axis = validate_axis(tensor.ndim, axis)

    op = TensorRechunk(
        chunk_size=None,
        axis=axis,
        factor=factor,
        num_partitions=num_partitions,
        rough=rough,
        dtype=tensor.dtype,
        sparse=tensor.issparse(),
    )
    return op(tensor)
