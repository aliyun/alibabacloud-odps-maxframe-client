# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

from typing import List

import numpy as np

from ... import opcodes
from ...core import ExecutableTuple
from ...lib.sparse.core import get_array_module
from ...serialization.serializables import AnyField, Int32Field
from ...typing_ import EntityType
from ..core import Tensor
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin
from ..utils import calc_sliced_size


class TensorSplit(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.ARRAY_SPLIT

    indices_or_sections = AnyField("indices_or_sections", default=None)
    axis = Int32Field("axis", default=None)

    @property
    def output_limit(self):
        return float("inf")

    @classmethod
    def _set_inputs(cls, op: "TensorSplit", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        if len(inputs) > 1:
            op.indices_or_sections = inputs[1]

    def __call__(self, a, indices_or_sections, is_split=False):
        axis = self.axis
        size = a.shape[axis]
        if np.isnan(size):
            raise ValueError(
                "cannot split array with unknown shape, "
                "call `.execute()` on input tensor first"
            )

        if (
            isinstance(indices_or_sections, Tensor)
            and hasattr(indices_or_sections.op, "data")
            and indices_or_sections.op.data is not None
        ):
            indices_or_sections = indices_or_sections.op.data

        try:
            indices_or_sections = int(indices_or_sections)
            if is_split:
                if size % indices_or_sections:
                    raise ValueError(
                        "tensor split does not result in an equal division"
                    )
                nparts = indices_or_sections
                nsplit = (size // indices_or_sections,) * nparts
            else:
                nparts = indices_or_sections
                if size % indices_or_sections == 0:
                    nsplit = (size // indices_or_sections,) * nparts
                else:
                    nsplit = (size // indices_or_sections + 1,) * (
                        size % indices_or_sections
                    ) + (size // indices_or_sections,) * (
                        size - size % indices_or_sections
                    )
        except TypeError:
            if isinstance(indices_or_sections, Tensor):
                nparts = indices_or_sections.shape[0] + 1
                nsplit = (np.nan,) * nparts
            else:
                ind = indices_or_sections = get_array_module(
                    indices_or_sections
                ).asarray(indices_or_sections)
                if indices_or_sections.ndim != 1 or not np.issubdtype(
                    indices_or_sections.dtype, np.integer
                ):
                    raise TypeError("slice indices must be integers or None")
                nparts = indices_or_sections.shape[0] + 1
                get = lambda i: None if i < 0 or i >= len(ind) else ind[i]
                nsplit = [
                    calc_sliced_size(size, slice(get(j - 1), get(j)))
                    for j in range(nparts)
                ]

        inputs = [a]
        if isinstance(indices_or_sections, Tensor):
            inputs.append(indices_or_sections)
        else:
            self.indices_or_sections = indices_or_sections

        kws = [
            {
                "i": i,
                "shape": a.shape[:axis] + (nsplit[i],) + a.shape[axis + 1 :],
                "order": a.order,
            }
            for i in range(nparts)
        ]
        return ExecutableTuple(self.new_tensors(inputs, kws=kws, output_limit=nparts))


def _split(a, indices_or_sections, axis=0, is_split=False):
    op = TensorSplit(axis=axis, dtype=a.dtype)
    return op(a, indices_or_sections, is_split=is_split)


def split(ary, indices_or_sections, axis=0):
    """
    Split a tensor into multiple sub-tensors.

    Parameters
    ----------
    ary : Tensor
        Tensor to be divided into sub-tensors.
    indices_or_sections : int or 1-D tensor
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal tensors along `axis`.  If such a split is not possible,
        an error is raised.

        If `indices_or_sections` is a 1-D tensor of sorted integers, the entries
        indicate where along `axis` the array is split.  For example,
        ``[2, 3]`` would, for ``axis=0``, result in

          - ary[:2]
          - ary[2:3]
          - ary[3:]

        If an index exceeds the dimension of the tensor along `axis`,
        an empty sub-tensor is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.

    Returns
    -------
    sub-tensors : list of Tensors
        A list of sub-tensors.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as an integer, but
        a split does not result in equal division.

    See Also
    --------
    array_split : Split a tensor into multiple sub-tensors of equal or
                  near-equal size.  Does not raise an exception if
                  an equal division cannot be made.
    hsplit : Split  into multiple sub-arrays horizontally (column-wise).
    vsplit : Split tensor into multiple sub-tensors vertically (row wise).
    dsplit : Split tensor into multiple sub-tensors along the 3rd axis (depth).
    concatenate : Join a sequence of tensors along an existing axis.
    stack : Join a sequence of tensors along a new axis.
    hstack : Stack tensors in sequence horizontally (column wise).
    vstack : Stack tensors in sequence vertically (row wise).
    dstack : Stack tensors in sequence depth wise (along third dimension).

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> x = mt.arange(9.0)
    >>> mt.split(x, 3).execute()
    [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.,  8.])]

    >>> x = mt.arange(8.0)
    >>> mt.split(x, [3, 5, 6, 10]).execute()
    [array([ 0.,  1.,  2.]),
     array([ 3.,  4.]),
     array([ 5.]),
     array([ 6.,  7.]),
     array([], dtype=float64)]

    """
    return _split(astensor(ary), indices_or_sections, axis=axis, is_split=True)
