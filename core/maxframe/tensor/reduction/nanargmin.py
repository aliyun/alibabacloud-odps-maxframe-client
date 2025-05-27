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
from ...serialization.serializables import AnyField, TupleField
from .core import TensorArgReductionMixin, TensorReduction


class TensorNanArgmin(TensorReduction, TensorArgReductionMixin):
    _op_type_ = opcodes.NANARGMIN
    _func_name = "nanargmin"
    _agg_func_name = "nanmin"

    offset = AnyField("offset", default=None)
    total_shape = TupleField("total_shape", default=None)

    def __init__(self, dtype=None, **kw):
        if dtype is None:
            dtype = np.dtype(int)
        super().__init__(dtype=dtype, **kw)


def nanargmin(a, axis=None, out=None):
    """
    Return the indices of the minimum values in the specified axis ignoring
    NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the results
    cannot be trusted if a slice contains only NaNs and Infs.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to operate. By default flattened input is used.

    Returns
    -------
    index_array : Tensor
        A tensor of indices or a single index value.

    See Also
    --------
    argmin, nanargmax

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> a = mt.array([[mt.nan, 4], [2, 3]])
    >>> mt.argmin(a).execute()
    0
    >>> mt.nanargmin(a).execute()
    2
    >>> mt.nanargmin(a, axis=0).execute()
    array([1, 1])
    >>> mt.nanargmin(a, axis=1).execute()
    array([1, 0])

    """
    op = TensorNanArgmin(axis=axis, dtype=np.dtype(int))
    return op(a, out=out)
