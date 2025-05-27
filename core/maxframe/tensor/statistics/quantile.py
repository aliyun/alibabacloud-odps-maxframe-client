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
from ...core import ENTITY_TYPE, EntityData
from ...serialization.serializables import AnyField, BoolField, KeyField, StringField
from ..core import TENSOR_TYPE, TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin
from ..utils import check_out_param


def _quantile_is_valid(q):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if q[i] < 0.0 or q[i] > 1.0:
                return False
    else:
        # faster than any()
        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
            return False
    return True


q_error_msg = "Quantiles must be in the range [0, 1]"


class TensorQuantile(TensorOperator, TensorOperatorMixin):
    __slots__ = ("q_error_msg",)
    _op_type_ = opcodes.QUANTILE

    a = KeyField("a")
    q = AnyField("q")
    axis = AnyField("axis")
    out = KeyField("out")
    overwrite_input = BoolField("overwrite_input")
    interpolation = StringField("interpolation")
    keepdims = BoolField("keepdims")

    def __init__(self, **kw):
        self.q_error_msg = kw.pop("q_error_msg", q_error_msg)
        super().__init__(**kw)

    @classmethod
    def _set_inputs(cls, op: "TensorQuantile", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.a = op._inputs[0]
        if isinstance(op.q, TENSOR_TYPE):
            op.q = op._inputs[1]
        if isinstance(op.out, TENSOR_TYPE):
            op.out = op._inputs[-1]

    def __call__(self, a, q=None, out=None):
        shape = [self.q.size] if self.q.ndim > 0 else []
        if self.axis is None:
            exclude_axes = set(range(a.ndim))
        elif isinstance(self.axis, tuple):
            exclude_axes = set(self.axis)
        else:
            exclude_axes = {self.axis}
        for ax, s in enumerate(a.shape):
            if ax not in exclude_axes:
                shape.append(s)
            elif self.keepdims:
                shape.append(1)
        inputs = [a] if q is None else [a, q]
        order = TensorOrder.C_ORDER
        if out is not None:
            inputs.append(out)
            order = out.order
            shape = out.shape
        t = self.new_tensor(inputs, shape=tuple(shape), order=order)
        if out is not None:
            check_out_param(out, t, "same_kind")
            out.data = t.data
            return out
        else:
            return t


INTERPOLATION_TYPES = {"linear", "lower", "higher", "midpoint", "nearest"}


def _quantile_unchecked(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    interpolation="linear",
    keepdims=False,
    q_error_msg=None,
    handle_non_numeric=None,
):
    a = astensor(a)
    raw_dtype = a.dtype
    need_view_back = False
    if handle_non_numeric and not np.issubdtype(a.dtype, np.number):
        # enable handle_non_numeric is often used
        # to handle the datetime-like dtype
        a = a.astype("i8")
        need_view_back = True
    if isinstance(q, ENTITY_TYPE):
        q = astensor(q)
        # do check in tile
        q_input = q
    else:
        q_input = None

    if isinstance(axis, Iterable):
        axis = tuple(axis)

    if q.ndim > 1:
        raise ValueError("`q` should be a scalar or array of float")

    if out is not None and not isinstance(out, TENSOR_TYPE):
        raise TypeError(f"`out` should be a tensor, got {type(out)}")

    if interpolation not in INTERPOLATION_TYPES:
        raise ValueError(
            "interpolation can only be 'linear', 'lower' "
            "'higher', 'midpoint', or 'nearest'"
        )

    # infer dtype
    q_tiny = np.random.rand(2 if q.size % 2 == 0 else 1).astype(q.dtype)
    if handle_non_numeric and not np.issubdtype(a.dtype, np.number):
        dtype = a.dtype
    else:
        dtype = np.quantile(
            np.empty(1, dtype=a.dtype), q_tiny, interpolation=interpolation
        ).dtype
    op = TensorQuantile(
        q=q,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        interpolation=interpolation,
        keepdims=keepdims,
        handle_non_numeric=handle_non_numeric,
        q_error_msg=q_error_msg,
        dtype=dtype,
        gpu=a.op.gpu,
    )
    ret = op(a, q=q_input, out=out)
    if need_view_back:
        ret = ret.astype(raw_dtype)
    return ret


def quantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    interpolation="linear",
    keepdims=False,
    **kw,
):
    """
    Compute the q-th quantile of the data along the specified axis.

    Parameters
    ----------
    a : array_like
        Input tensor or object that can be converted to a tensor.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The
        default is to compute the quantile(s) along a flattened
        version of the tensor.
    out : Tensor, optional
        Alternative output tensor in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        Just for compatibility with Numpy, would not take effect.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to
        use when the desired quantile lies between two data points
        ``i < j``:

            * linear: ``i + (j - i) * fraction``, where ``fraction``
              is the fractional part of the index surrounded by ``i``
              and ``j``.
            * lower: ``i``.
            * higher: ``j``.
            * nearest: ``i`` or ``j``, whichever is nearest.
            * midpoint: ``(i + j) / 2``.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original tensor `a`.

    Returns
    -------
    quantile : scalar or Tensor
        If `q` is a single quantile and `axis=None`, then the result
        is a scalar. If multiple quantiles are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that tensor is
        returned instead.

    See Also
    --------
    mean
    percentile : equivalent to quantile, but with q in the range [0, 100].
    median : equivalent to ``quantile(..., 0.5)``
    nanquantile

    Notes
    -----
    Given a vector ``V`` of length ``N``, the q-th quantile of
    ``V`` is the value ``q`` of the way from the minimum to the
    maximum in a sorted copy of ``V``. The values and distances of
    the two nearest neighbors as well as the `interpolation` parameter
    will determine the quantile if the normalized ranking does not
    match the location of ``q`` exactly. This function is the same as
    the median if ``q=0.5``, the same as the minimum if ``q=0.0`` and the
    same as the maximum if ``q=1.0``.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> a = mt.array([[10, 7, 4], [3, 2, 1]])
    >>> a.execute()
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> mt.quantile(a, 0.5).execute()
    3.5
    >>> mt.quantile(a, 0.5, axis=0).execute()
    array([6.5, 4.5, 2.5])
    >>> mt.quantile(a, 0.5, axis=1).execute()
    array([7.,  2.])
    >>> mt.quantile(a, 0.5, axis=1, keepdims=True).execute()
    array([[7.],
           [2.]])
    >>> m = mt.quantile(a, 0.5, axis=0)
    >>> out = mt.zeros_like(m)
    >>> mt.quantile(a, 0.5, axis=0, out=out).execute()
    array([6.5, 4.5, 2.5])
    >>> m.execute()
    array([6.5, 4.5, 2.5])
    """

    handle_non_numeric = kw.pop("handle_non_numeric", None)
    if len(kw) > 0:  # pragma: no cover
        raise TypeError(
            f"quantile() got an unexpected keyword argument '{next(iter(kw))}'"
        )

    if not isinstance(q, ENTITY_TYPE):
        q = np.asanyarray(q)
        # do check instantly if q is not a tensor
        if not _quantile_is_valid(q):
            raise ValueError(q_error_msg)

    return _quantile_unchecked(
        a,
        q,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        interpolation=interpolation,
        keepdims=keepdims,
        handle_non_numeric=handle_non_numeric,
    )
