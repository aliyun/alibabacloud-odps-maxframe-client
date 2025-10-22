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
from pandas.api.types import is_list_like

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData, get_output_types
from ...serialization.serializables import AnyField
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index, validate_axis


class DataFrameRepeat(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.REPEAT

    repeats = AnyField("repeats", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @classmethod
    def _set_inputs(cls, op: "DataFrameRepeat", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if isinstance(op.repeats, ENTITY_TYPE):
            op.repeats = inputs[1]

    def __call__(self, obj, repeats):
        self._output_types = get_output_types(obj)
        test_index = obj.index_value.to_pandas()[:0]

        params = obj.params
        params["index_value"] = parse_index(test_index, obj, type(self), self.repeats)
        params["shape"] = (np.nan,)

        inputs = [obj]
        if isinstance(repeats, ENTITY_TYPE):
            inputs.append(repeats)
        return self.new_tileable(inputs, **params)


def _repeat(obj, repeats, axis=None):
    from ...tensor.datasource import tensor

    axis = validate_axis(axis or 0, obj)
    if is_list_like(repeats):
        repeats = tensor(repeats)
    op = DataFrameRepeat(repeats=repeats, axis=axis)
    return op(obj, repeats)


def series_repeat(obj, repeats, axis=None):
    """
    Repeat elements of a Series.

    Returns a new Series where each element of the current Series
    is repeated consecutively a given number of times.

    Parameters
    ----------
    repeats : int or array of ints
        The number of repetitions for each element. This should be a
        non-negative integer. Repeating 0 times will return an empty
        Series.
    axis : None
        Must be ``None``. Has no effect but is accepted for compatibility
        with numpy.

    Returns
    -------
    Series
        Newly created Series with repeated elements.

    See Also
    --------
    Index.repeat : Equivalent function for Index.
    numpy.repeat : Similar method for :class:`numpy.ndarray`.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s = md.Series(['a', 'b', 'c'])
    >>> s.execute()
    0    a
    1    b
    2    c
    dtype: object
    >>> s.repeat(2).execute()
    0    a
    0    a
    1    b
    1    b
    2    c
    2    c
    dtype: object
    >>> s.repeat([1, 2, 3]).execute()
    0    a
    1    b
    1    b
    2    c
    2    c
    2    c
    dtype: object
    """
    return _repeat(obj, repeats, axis=axis)


def index_repeat(obj, repeats, axis=None):
    """
    Repeat elements of an Index.

    Returns a new Index where each element of the current Index
    is repeated consecutively a given number of times.

    Parameters
    ----------
    repeats : int or array of ints
        The number of repetitions for each element. This should be a
        non-negative integer. Repeating 0 times will return an empty
        Index.
    axis : None
        Must be ``None``. Has no effect but is accepted for compatibility
        with numpy.

    Returns
    -------
    repeated_index : Index
        Newly created Index with repeated elements.

    See Also
    --------
    Series.repeat : Equivalent function for Series.
    numpy.repeat : Similar method for :class:`numpy.ndarray`.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> idx = md.Index(['a', 'b', 'c'])
    >>> idx.execute()
    Index(['a', 'b', 'c'], dtype='object')
    >>> idx.repeat(2).execute()
    Index(['a', 'a', 'b', 'b', 'c', 'c'], dtype='object')
    >>> idx.repeat([1, 2, 3]).execute()
    Index(['a', 'b', 'b', 'c', 'c', 'c'], dtype='object')
    """
    return _repeat(obj, repeats, axis=axis)
