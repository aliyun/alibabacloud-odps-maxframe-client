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
from ...core import TILEABLE_TYPE, EntityData
from ...serialization.serializables import AnyField, Int32Field, Int64Field, StringField
from ..core import TENSOR_TYPE, TensorOrder
from ..datasource.array import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin


class TensorSearchsorted(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.SEARCHSORTED

    a = AnyField("a")
    v = AnyField("v")
    side = StringField("side")
    combine_size = Int32Field("combine_size")
    # for chunk
    offset = Int64Field("offset")
    size = Int64Field("size")
    n_chunk = Int64Field("n_chunk")

    @classmethod
    def _set_inputs(cls, op: "TensorSearchsorted", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.a = inputs[0]
        if isinstance(op.v, TILEABLE_TYPE):
            op.v = op._inputs[1]

    def __call__(self, a, v):
        inputs = [a]
        if isinstance(v, TILEABLE_TYPE):
            inputs.append(v)
            shape = v.shape
        else:
            shape = ()
        return self.new_tensor(inputs, shape=shape, order=TensorOrder.C_ORDER)


def searchsorted(a, v, side="left", sorter=None, combine_size=None):
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted tensor `a` such that, if the
    corresponding elements in `v` were inserted before the indices, the
    order of `a` would be preserved.

    Assuming that `a` is sorted:

    ======  ============================
    `side`  returned index `i` satisfies
    ======  ============================
    left    ``a[i-1] < v <= a[i]``
    right   ``a[i-1] <= v < a[i]``
    ======  ============================

    Parameters
    ----------
    a : 1-D array_like
        Input tensor. If `sorter` is None, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        that sort it.
    v : array_like
        Values to insert into `a`.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.  If there is no suitable
        index, return either 0 or N (where N is the length of `a`).
    sorter : 1-D array_like, optional
        Optional tensor of integer indices that sort array a into ascending
        order. They are typically the result of argsort.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    indices : tensor of ints
        Array of insertion points with the same shape as `v`.

    See Also
    --------
    sort : Return a sorted copy of a tensor.
    histogram : Produce histogram from 1-D data.

    Notes
    -----
    Binary search is used to find the required insertion points.

    This function is a faster version of the builtin python `bisect.bisect_left`
    (``side='left'``) and `bisect.bisect_right` (``side='right'``) functions,
    which is also vectorized in the `v` argument.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.searchsorted([1,2,3,4,5], 3).execute()
    2
    >>> mt.searchsorted([1,2,3,4,5], 3, side='right').execute()
    3
    >>> mt.searchsorted([1,2,3,4,5], [-10, 10, 2, 3]).execute()
    array([0, 5, 1, 2])

    """

    if (
        not isinstance(a, TENSOR_TYPE)
        and sorter is not None
        and not isinstance(sorter, TENSOR_TYPE)
    ):
        a = astensor(np.asarray(a)[sorter])
    else:
        a = astensor(a)
        if sorter is not None:
            a = a[sorter]

    if a.ndim != 1:
        raise ValueError("`a` should be 1-d tensor")
    if a.issparse():
        # does not support sparse tensor
        raise ValueError("`a` should be a dense tensor")
    if side not in {"left", "right"}:
        raise ValueError(f"'{side}' is an invalid value for keyword 'side'")

    if not np.isscalar(v):
        v = astensor(v)

    op = TensorSearchsorted(
        v=v, side=side, dtype=np.dtype(np.intp), combine_size=combine_size
    )
    return op(a, v)
