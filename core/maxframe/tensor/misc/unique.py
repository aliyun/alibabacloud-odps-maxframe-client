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

import numpy as np

from ... import opcodes
from ...serialization.serializables import BoolField, Int32Field
from ..core import TensorOrder
from ..operators import TensorHasInput, TensorOperatorMixin
from ..utils import validate_axis


class TensorUnique(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.UNIQUE

    return_index = BoolField("return_index", default=False)
    return_inverse = BoolField("return_inverse", default=False)
    return_counts = BoolField("return_counts", default=False)
    axis = Int32Field("axis", default=None)

    @property
    def output_limit(self):
        return 1 + self.return_index + self.return_inverse + self.return_counts

    @classmethod
    def _gen_kws(cls, op: "TensorUnique", input_obj, chunk=False, chunk_index=None):
        kws = []

        # unique tensor
        shape = list(input_obj.shape)
        shape[op.axis] = np.nan
        kw = {"shape": tuple(shape), "dtype": input_obj.dtype, "gpu": input_obj.op.gpu}
        if chunk:
            idx = [0] * len(shape)
            idx[op.axis] = chunk_index or 0
            kw["index"] = tuple(idx)
        kws.append(kw)

        # unique indices tensor
        if op.return_index:
            kw = {
                "shape": (np.nan,),
                "dtype": np.dtype(np.intp),
                "gpu": input_obj.op.gpu,
                "type": "indices",
            }
            if chunk:
                kw["index"] = (chunk_index or 0,)
            kws.append(kw)

        # unique inverse tensor
        if op.return_inverse:
            kw = {
                "shape": (input_obj.shape[op.axis],),
                "dtype": np.dtype(np.intp),
                "gpu": input_obj.op.gpu,
                "type": "inverse",
            }
            if chunk:
                kw["index"] = (chunk_index or 0,)
            kws.append(kw)

        # unique counts tensor
        if op.return_counts:
            kw = {
                "shape": (np.nan,),
                "dtype": np.dtype(int),
                "gpu": input_obj.op.gpu,
                "type": "counts",
            }
            if chunk:
                kw["index"] = (chunk_index or 0,)
            kws.append(kw)

        return kws

    def __call__(self, ar):
        from .atleast_1d import atleast_1d

        ar = atleast_1d(ar)
        if self.axis is None:
            if ar.ndim > 1:
                ar = ar.flatten()
            self.axis = 0
        else:
            self.axis = validate_axis(ar.ndim, self.axis)

        kws = self._gen_kws(self, ar)
        tensors = self.new_tensors([ar], kws=kws, order=TensorOrder.C_ORDER)
        if len(tensors) == 1:
            return tensors[0]
        return tensors


def unique(
    ar,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    axis=None,
):
    """
    Find the unique elements of a tensor.

    Returns the sorted unique elements of a tensor. There are three optional
    outputs in addition to the unique elements:

    * the indices of the input tensor that give the unique values
    * the indices of the unique tensor that reconstruct the input tensor
    * the number of times each unique value comes up in the input tensor

    Parameters
    ----------
    ar : array_like
        Input tensor. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened tensor) that result in the unique tensor.
    return_inverse : bool, optional
        If True, also return the indices of the unique tensor (for the specified
        axis, if provided) that can be used to reconstruct `ar`.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.
    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened. If an integer,
        the subarrays indexed by the given axis will be flattened and treated
        as the elements of a 1-D tensor with the dimension of the given axis,
        see the notes for more details.  Object tensors or structured tensors
        that contain objects are not supported if the `axis` kwarg is used. The
        default is None.

    Returns
    -------
    unique : Tensor
        The sorted unique values.
    unique_indices : Tensor, optional
        The indices of the first occurrences of the unique values in the
        original tensor. Only provided if `return_index` is True.
    unique_inverse : Tensor, optional
        The indices to reconstruct the original tensor from the
        unique tensor. Only provided if `return_inverse` is True.
    unique_counts : Tensor, optional
        The number of times each of the unique values comes up in the
        original tensor. Only provided if `return_counts` is True.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.unique([1, 1, 2, 2, 3, 3]).execute()
    array([1, 2, 3])
    >>> a = mt.array([[1, 1], [2, 3]])
    >>> mt.unique(a).execute()
    array([1, 2, 3])

    Return the unique rows of a 2D tensor

    >>> a = mt.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> mt.unique(a, axis=0).execute()
    array([[1, 0, 0], [2, 3, 4]])

    Return the indices of the original tensor that give the unique values:

    >>> a = mt.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = mt.unique(a, return_index=True)
    >>> u.execute()
    array(['a', 'b', 'c'],
           dtype='|S1')
    >>> indices.execute()
    array([0, 1, 3])
    >>> a[indices].execute()
    array(['a', 'b', 'c'],
           dtype='|S1')

    Reconstruct the input array from the unique values:

    >>> a = mt.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = mt.unique(a, return_inverse=True)
    >>> u.execute()
    array([1, 2, 3, 4, 6])
    >>> indices.execute()
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices].execute()
    array([1, 2, 6, 4, 2, 3, 2])
    """
    op = TensorUnique(
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
    )
    return op(ar)
