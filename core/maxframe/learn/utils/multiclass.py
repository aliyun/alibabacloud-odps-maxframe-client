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

from collections.abc import Sequence
from typing import List

import numpy as np
from scipy.sparse import spmatrix

from ... import opcodes
from ... import tensor as mt
from ...core import ENTITY_TYPE, TILEABLE_TYPE, OutputType
from ...core.operator import Operator
from ...serialization.serializables import AnyField, ListField
from ...tensor.core import TENSOR_TYPE, TensorOrder
from ...typing_ import EntityType, TileableType
from ...udf import builtin_function
from ..core import LearnOperatorMixin


class UniqueLabels(Operator, LearnOperatorMixin):
    _op_type_ = opcodes.UNIQUE_LABELS

    ys = ListField("ys")

    def __call__(self, ys: List[TileableType]):
        self._output_types = [OutputType.tensor]
        inputs = [y for y in ys if isinstance(y, TILEABLE_TYPE)]
        return self.new_tileable(
            inputs,
            shape=(np.nan,),
            dtype=mt.tensor(ys[0]).dtype,
            order=TensorOrder.C_ORDER,
        )


def unique_labels(*ys):
    """
    Extract an ordered array of unique labels.

    We don't allow:
        - mix of multilabel and multiclass (single label) targets
        - mix of label indicator matrix and anything else,
          because there are no explicit labels)
        - mix of label indicator matrices of different sizes
        - mix of string and integer labels

    At the moment, we also don't allow "multiclass-multioutput" input type.

    Parameters
    ----------
    *ys : array-likes

    Returns
    -------
    out : ndarray of shape (n_unique_labels,)
        An ordered array of unique labels.

    Examples
    --------
    >>> from maxframe.learn.utils.multiclass import unique_labels
    >>> unique_labels([3, 5, 5, 5, 7, 7]).execute()
    array([3, 5, 7])
    >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4]).execute()
    array([1, 2, 3, 4])
    >>> unique_labels([1, 2, 10], [5, 11]).execute()
    array([ 1,  2,  5, 10, 11])
    """
    if not ys:
        raise ValueError("No argument has been passed.")

    ys = list(ys)
    op = UniqueLabels(ys=ys)
    return op(ys)


class IsMultilabel(Operator, LearnOperatorMixin):
    _op_type_ = opcodes.IS_MULTILABEL

    y = AnyField("y")

    def __call__(self, y):
        self._output_types = [OutputType.tensor]
        inputs = [y] if isinstance(y, ENTITY_TYPE) else []
        return self.new_tileable(
            inputs, shape=(), dtype=np.dtype(bool), order=TensorOrder.C_ORDER
        )

    @classmethod
    def _set_inputs(cls, op: "IsMultilabel", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        if op._inputs:
            op.y = op._inputs[0]


def is_multilabel(y):
    """
    Check if ``y`` is in a multilabel format.

    Parameters
    ----------
    y : numpy array of shape [n_samples]
        Target values.

    Returns
    -------
    out : bool,
        Return ``True``, if ``y`` is in a multilabel format, else ```False``.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> from maxframe.learn.utils.multiclass import is_multilabel
    >>> is_multilabel([0, 1, 0, 1]).execute()
    False
    >>> is_multilabel([[1], [0, 2], []]).execute()
    False
    >>> is_multilabel(mt.array([[1, 0], [0, 0]])).execute()
    True
    >>> is_multilabel(mt.array([[1], [0], [0]])).execute()
    False
    >>> is_multilabel(mt.array([[1, 0, 0]])).execute()
    True
    """
    if not isinstance(y, ENTITY_TYPE):
        if hasattr(y, "__array__") or isinstance(y, Sequence):
            y = np.asarray(y)
        yt = None
    else:
        yt = y = mt.tensor(y)

    op = IsMultilabel(y=y)
    return op(yt)


class TypeOfTarget(Operator, LearnOperatorMixin):
    _op_type_ = opcodes.TYPE_OF_TARGET

    y = AnyField("y")

    def __call__(self, y: TileableType):
        self._output_types = [OutputType.tensor]
        inputs = [y] if isinstance(y, ENTITY_TYPE) else []
        return self.new_tileable(
            inputs, shape=(), order=TensorOrder.C_ORDER, dtype=np.dtype(object)
        )

    @classmethod
    def _set_inputs(cls, op: "TypeOfTarget", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        if op._inputs:
            op.y = op._inputs[0]


def type_of_target(y):
    """
    Determine the type of data indicated by the target.

    Note that this type is the most specific type that can be inferred.
    For example:

        * ``binary`` is more specific but compatible with ``multiclass``.
        * ``multiclass`` of integers is more specific but compatible with
          ``continuous``.
        * ``multilabel-indicator`` is more specific but compatible with
          ``multiclass-multioutput``.

    Parameters
    ----------
    y : array-like

    Returns
    -------
    target_type : string
        One of:

        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d tensor of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `y` is a 2d tensor that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `y` is a label indicator matrix, a tensor
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          tensor, sequence of sequences, or a tensor of non-sequence objects.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> from maxframe.learn.utils.multiclass import type_of_target
    >>> type_of_target([0.1, 0.6]).execute()
    'continuous'
    >>> type_of_target([1, -1, -1, 1]).execute()
    'binary'
    >>> type_of_target(['a', 'b', 'a']).execute()
    'binary'
    >>> type_of_target([1.0, 2.0]).execute()
    'binary'
    >>> type_of_target([1, 0, 2]).execute()
    'multiclass'
    >>> type_of_target([1.0, 0.0, 3.0]).execute()
    'multiclass'
    >>> type_of_target(['a', 'b', 'c']).execute()
    'multiclass'
    >>> type_of_target(mt.array([[1, 2], [3, 1]])).execute()
    'multiclass-multioutput'
    >>> type_of_target([[1, 2]]).execute()
    'multiclass-multioutput'
    >>> type_of_target(mt.array([[1.5, 2.0], [3.0, 1.6]])).execute()
    'continuous-multioutput'
    >>> type_of_target(mt.array([[0, 1], [1, 1]])).execute()
    'multilabel-indicator'
    """
    if isinstance(y, TENSOR_TYPE):
        y = mt.tensor(y)

    valid_types = (Sequence, spmatrix) if spmatrix is not None else (Sequence,)
    valid = (
        isinstance(y, valid_types)
        or hasattr(y, "__array__")
        or hasattr(y, "__maxframe_tensor__")
    ) and not isinstance(y, str)

    if not valid:
        raise ValueError(f"Expected array-like (array or non-string sequence), got {y}")

    sparse_pandas = type(y).__name__ in ["SparseSeries", "SparseArray"]
    if sparse_pandas:  # pragma: no cover
        raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")

    if isinstance(y, ENTITY_TYPE):
        y = mt.tensor(y)

    op = TypeOfTarget(y=y)
    return op(y)


@builtin_function
def _check_class_target_name_and_return(t, t_type=None):
    t_type = t_type or t
    if t_type not in [
        "binary",
        "multiclass",
        "multiclass-multioutput",
        "multilabel-indicator",
        "multilabel-sequences",
    ]:
        raise ValueError("Unknown label type: %r" % t_type)
    return t


def check_classification_targets(y, return_value: bool = False):
    """
    Ensure that target y is of a non-regression type.

    Only the following target types (as defined in type_of_target) are allowed:
        'binary', 'multiclass', 'multiclass-multioutput',
        'multilabel-indicator', 'multilabel-sequences'

    Parameters
    ----------
    y : array-like
    """
    y_type = type_of_target(y)

    y_type = y_type.mf.apply_chunk(
        _check_class_target_name_and_return, dtype=y_type.dtype
    )
    if not return_value:
        return y_type
    y = mt.array(y)
    return y_type, y.mf.apply_chunk(
        _check_class_target_name_and_return, args=(y_type,), **y.params
    )
