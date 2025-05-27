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
from typing import List

import numpy as np

from ... import opcodes
from ...core import EntityData
from ...serialization.serializables import FieldTypes, KeyField, TupleField
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin


class TensorTensorDot(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.TENSORDOT

    a = KeyField("a")
    b = KeyField("b")
    a_axes = TupleField("a_axes", FieldTypes.int32)
    b_axes = TupleField("b_axes", FieldTypes.int32)

    @classmethod
    def _set_inputs(cls, op: "TensorTensorDot", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.a = op._inputs[0]
        op.b = op._inputs[1]

    def __call__(self, a, b):
        shape = tuple(
            s for i, s in enumerate(a.shape) if i not in set(self.a_axes)
        ) + tuple(s for i, s in enumerate(b.shape) if i not in set(self.b_axes))
        return self.new_tensor([a, b], shape, order=TensorOrder.C_ORDER)


def tensordot(a, b, axes=2, sparse=None):
    """
    Compute tensor dot product along specified axes for tensors >= 1-D.

    Given two tensors (arrays of dimension greater than or equal to one),
    `a` and `b`, and an array_like object containing two array_like
    objects, ``(a_axes, b_axes)``, sum the products of `a`'s and `b`'s
    elements (components) over the axes specified by ``a_axes`` and
    ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N``
    dimensions of `a` and the first ``N`` dimensions of `b` are summed
    over.

    Parameters
    ----------
    a, b : array_like, len(shape) >= 1
        Tensors to "dot".
    axes : int or (2,) array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.

    See Also
    --------
    dot, einsum

    Notes
    -----
    Three common use cases are:

        * ``axes = 0`` : tensor product :math:`a\\otimes b`
        * ``axes = 1`` : tensor dot product :math:`a\\cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`

    When `axes` is integer_like, the sequence for evaluation will be: first
    the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
    Nth axis in `b` last.

    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.

    Examples
    --------
    >>> import maxframe.tensor as mt

    A "traditional" example:

    >>> a = mt.arange(60.).reshape(3,4,5)
    >>> b = mt.arange(24.).reshape(4,3,2)
    >>> c = mt.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)

    >>> r = c.execute()
    >>> r
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])

    >>> # A slower but equivalent way of computing the same...
    >>> ra = np.arange(60.).reshape(3,4,5)
    >>> rb = np.arange(24.).reshape(4,3,2)
    >>> d = np.zeros((5,2))
    >>> for i in range(5):
    ...   for j in range(2):
    ...     for k in range(3):
    ...       for n in range(4):
    ...         d[i,j] += ra[k,n,i] * rb[n,k,j]
    >>> r == d
    array([[ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True]], dtype=bool)

    An extended example taking advantage of the overloading of + and \\*:

    >>> a = mt.array(range(1, 9))
    >>> a.shape = (2, 2, 2)
    >>> A = mt.array(('a', 'b', 'c', 'd'), dtype=object)
    >>> A.shape = (2, 2)
    >>> a.execute(); A.execute()
    array([[[1, 2],
            [3, 4]],
           [[5, 6],
            [7, 8]]])
    array([[a, b],
           [c, d]], dtype=object)

    >>> mt.tensordot(a, A).execute() # third argument default is 2 for double-contraction
    array([abbcccdddd, aaaaabbbbbbcccccccdddddddd], dtype=object)

    >>> mt.tensordot(a, A, 1).execute()
    array([[[acc, bdd],
            [aaacccc, bbbdddd]],
           [[aaaaacccccc, bbbbbdddddd],
            [aaaaaaacccccccc, bbbbbbbdddddddd]]], dtype=object)

    >>> mt.tensordot(a, A, 0).execute() # tensor product (result too long to incl.)
    array([[[[[a, b],
              [c, d]],
              ...

    >>> mt.tensordot(a, A, (0, 1)).execute()
    array([[[abbbbb, cddddd],
            [aabbbbbb, ccdddddd]],
           [[aaabbbbbbb, cccddddddd],
            [aaaabbbbbbbb, ccccdddddddd]]], dtype=object)

    >>> mt.tensordot(a, A, (2, 1)).execute()
    array([[[abb, cdd],
            [aaabbbb, cccdddd]],
           [[aaaaabbbbbb, cccccdddddd],
            [aaaaaaabbbbbbbb, cccccccdddddddd]]], dtype=object)

    >>> mt.tensordot(a, A, ((0, 1), (0, 1))).execute()
    array([abbbcccccddddddd, aabbbbccccccdddddddd], dtype=object)

    >>> mt.tensordot(a, A, ((2, 1), (1, 0))).execute()
    array([acccbbdddd, aaaaacccccccbbbbbbdddddddd], dtype=object)
    """
    a = astensor(a)
    b = astensor(b)

    if isinstance(axes, Iterable):
        a_axes, b_axes = axes
    else:
        a_axes = tuple(range(a.ndim - 1, a.ndim - axes - 1, -1))
        b_axes = tuple(range(0, axes))

    if isinstance(a_axes, Iterable):
        a_axes = tuple(a_axes)
    else:
        a_axes = (a_axes,)
    a_axes = tuple(axis if axis >= 0 else a.ndim + axis for axis in a_axes)
    if isinstance(b_axes, Iterable):
        b_axes = tuple(b_axes)
    else:
        b_axes = (b_axes,)
    b_axes = tuple(axis if axis >= 0 else b.ndim + axis for axis in b_axes)

    if (
        a.shape
        and b.shape
        and not np.array_equal(
            np.array(a.shape)[list(a_axes)], np.array(b.shape)[list(b_axes)]
        )
    ):
        raise ValueError("shape-mismatch for sum")

    sparse = sparse if sparse is not None else a.issparse() and b.issparse()
    op = TensorTensorDot(
        a_axes=a_axes,
        b_axes=b_axes,
        dtype=np.promote_types(a.dtype, b.dtype),
        sparse=sparse,
    )
    return op(a, b)
