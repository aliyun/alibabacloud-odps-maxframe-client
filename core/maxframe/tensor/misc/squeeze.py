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

from collections.abc import Iterable

from ... import opcodes
from ...serialization.serializables import FieldTypes, TupleField
from ..operators import TensorHasInput, TensorOperatorMixin


class TensorSqueeze(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.SQUEEZE

    axis = TupleField("axis", FieldTypes.int32)

    def on_output_modify(self, new_output):
        slcs = [slice(None)] * new_output.ndim
        for axis in self.axis:
            slcs.insert(axis, None)
        return new_output[slcs]

    def on_input_modify(self, new_input):
        op = self.copy().reset_key()
        return op(new_input, self.outputs[0].shape)

    @staticmethod
    def _get_squeeze_shape(shape, axis):
        if axis is not None:
            if isinstance(axis, Iterable):
                axis = tuple(axis)
            else:
                axis = (axis,)

            for ax in axis:
                if shape[ax] != 1:
                    raise ValueError(
                        "cannot select an axis to squeeze out "
                        "which has size not equal to one"
                    )
            shape = tuple(s for i, s in enumerate(shape) if i not in axis)
        else:
            axis = tuple(i for i, s in enumerate(shape) if s == 1)
            shape = tuple(s for s in shape if s != 1)

        return shape, axis

    def __call__(self, a, shape):
        return self.new_tensor([a], shape, order=a.order)


def squeeze(a, axis=None):
    """
    Remove single-dimensional entries from the shape of a tensor.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Selects a subset of the single-dimensional entries in the
        shape. If an axis is selected with shape entry greater than
        one, an error is raised.

    Returns
    -------
    squeezed : Tensor
        The input tensor, but with all or a subset of the
        dimensions of length 1 removed. This is always `a` itself
        or a view into `a`.

    Raises
    ------
    ValueError
        If `axis` is not `None`, and an axis being squeezed is not of length 1

    See Also
    --------
    expand_dims : The inverse operation, adding singleton dimensions
    reshape : Insert, remove, and combine dimensions, and resize existing ones

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> x = mt.array([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> mt.squeeze(x).shape
    (3,)
    >>> mt.squeeze(x, axis=0).shape
    (3, 1)
    >>> mt.squeeze(x, axis=1).shape
    Traceback (most recent call last):
    ...
    ValueError: cannot select an axis to squeeze out which has size not equal to one
    >>> mt.squeeze(x, axis=2).shape
    (1, 3)

    """
    shape, axis = TensorSqueeze._get_squeeze_shape(a.shape, axis)

    if 1 not in a.shape:
        return a

    op = TensorSqueeze(axis=axis, dtype=a.dtype, sparse=a.issparse())
    return op(a, shape)
