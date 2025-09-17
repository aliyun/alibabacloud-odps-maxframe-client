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

from numbers import Integral
from typing import List

import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import find_common_type
from pandas.core.indexing import IndexingError

from ... import opcodes
from ...config import options
from ...core import ENTITY_TYPE, EntityData, OutputType
from ...serialization.serializables import AnyField, KeyField, ListField
from ...tensor import asarray
from ...tensor.indexing.core import calc_shape
from ..operators import DATAFRAME_TYPE, DataFrameOperator, DataFrameOperatorMixin
from ..utils import indexing_index_value, validate_axis

_ILOC_ERROR_MSG = (
    "Location based indexing can only have [integer, "
    "integer slice (START point is INCLUDED, END point is EXCLUDED), "
    "listlike of integers, boolean array] types"
)


def process_iloc_indexes(inp, indexes, axis=0):
    ndim = inp.ndim

    if not isinstance(indexes, tuple):
        indexes = (indexes,)
    if len(indexes) < ndim:
        if not axis:
            indexes += (slice(None),) * (ndim - len(indexes))
        else:
            indexes = (slice(None),) * (ndim - len(indexes)) + indexes
    if len(indexes) > ndim:
        raise IndexingError("Too many indexers")

    new_indexes = []
    # check each index
    for ax, index in enumerate(indexes):
        if isinstance(index, tuple):
            # a tuple should already have been caught by this point
            # so don't treat a tuple as a valid indexer
            raise IndexingError("Too many indexers")
        elif isinstance(index, slice):
            if any(v is not None for v in [index.start, index.stop, index.step]):
                pd_index = (
                    inp.index_value if ax == 0 else inp.columns_value
                ).to_pandas()
                for val in [index.start, index.stop, index.step]:
                    if val is not None:
                        try:
                            pd_index[val]  # check on the pandas
                        except IndexError:
                            pass
                        except TypeError:
                            raise TypeError(
                                f"cannot do slice indexing on {type(pd_index)} "
                                f"with these indexers [{val}] of {type(val)}"
                            )
            new_indexes.append(index)
        elif isinstance(index, (list, np.ndarray, pd.Series, ENTITY_TYPE)):
            if not isinstance(index, ENTITY_TYPE):
                index = np.asarray(index)
            else:
                index = asarray(index)
                if ax == 1:
                    # do not support tensor index on axis 1
                    # because if so, the dtypes and columns_value would be unknown
                    try:
                        index = index.fetch()
                    except (RuntimeError, ValueError):
                        raise NotImplementedError(
                            "indexer on axis columns cannot be non-executed tensor"
                        )
            if index.dtype != np.bool_:
                index = index.astype(np.int64)
            if index.ndim != 1:
                raise ValueError(
                    "Buffer has wrong number of dimensions "
                    f"(expected 1, got {index.ndim})"
                )
            new_indexes.append(index)
        elif isinstance(index, Integral):
            shape = inp.shape[ax]
            if not np.isnan(shape):
                if index < -shape or index >= shape:
                    raise IndexError("single positional indexer is out-of-bounds")
            new_indexes.append(index)
        else:
            raise ValueError(_ILOC_ERROR_MSG)

    return new_indexes


class DataFrameIloc:
    def __init__(self, obj, axis=None):
        self._obj = obj
        self._axis = axis

    def __getitem__(self, indexes):
        indexes = process_iloc_indexes(self._obj, indexes, axis=self._axis)
        if isinstance(self._obj, DATAFRAME_TYPE):
            op = DataFrameIlocGetItem(indexes=indexes)
        else:
            op = SeriesIlocGetItem(indexes=indexes)
        return op(self._obj)

    def __setitem__(self, indexes, value):
        if not np.isscalar(value):
            raise NotImplementedError("Only scalar value is supported to set by iloc")

        indexes = process_iloc_indexes(self._obj, indexes, axis=self._axis)
        if isinstance(self._obj, DATAFRAME_TYPE):
            op = DataFrameIlocSetItem(indexes=indexes, value=value)
        else:
            op = SeriesIlocSetItem(indexes=indexes, value=value)

        ret = op(self._obj)
        self._obj.data = ret.data

    def __call__(self, axis):
        axis = validate_axis(axis, self._obj)
        return DataFrameIloc(self._obj, axis)


class HeadTailOptimizedOperatorMixin(DataFrameOperatorMixin):
    __slots__ = ()

    @classmethod
    def _is_head(cls, index0):
        return (
            (index0.start is None or index0.start == 0)
            and index0.stop is not None
            and index0.stop > 0
        )

    @classmethod
    def _is_tail(cls, index0):
        return index0.start is not None and index0.start < 0 and index0.stop is None

    @classmethod
    def _is_indexes_head_or_tail(cls, indexes):
        index0 = indexes[0]
        if not isinstance(index0, slice):
            # have to be slice
            return False
        if index0.step is not None and index0.step != 1:
            return False
        if len(indexes) == 2:
            if not isinstance(indexes[1], slice):
                return False
            if indexes[1] != slice(None):
                return False
        if cls._is_tail(index0):
            # tail
            return True
        if cls._is_head(index0):
            # head
            return True
        return False

    def can_be_optimized(self):
        return (
            self._is_indexes_head_or_tail(self.indexes)
            and self._is_head(self.indexes[0])
            and self.indexes[0].stop <= options.optimize.head_optimize_threshold
        )


class DataFrameIlocGetItem(DataFrameOperator, HeadTailOptimizedOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_ILOC_GETITEM

    _input = KeyField("input")
    indexes = ListField("indexes", default=None)

    def __init__(self, gpu=None, sparse=False, **kw):
        if kw.get("output_types"):
            kw["_output_types"] = kw.pop("output_types")
        super().__init__(gpu=gpu, sparse=sparse, **kw)
        if not self.output_types:
            self._output_types = [OutputType.dataframe]

    @property
    def input(self):
        return self._input

    @classmethod
    def _set_inputs(cls, op: "DataFrameIlocGetItem", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs)
        op._input = next(inputs_iter)
        indexes = []
        for index in op.indexes:
            if isinstance(index, ENTITY_TYPE):
                indexes.append(next(inputs_iter))
            else:
                indexes.append(index)
        op.indexes = indexes

    def __call__(self, df):
        # Note [Fancy Index of Numpy and Pandas]
        #
        # The numpy and pandas.iloc have different semantic when processing fancy index:
        #
        # >>> np.ones((3,3))[[1,2],[1,2]]
        # array([1., 1.])
        #
        # >>> pd.DataFrame(np.ones((3,3))).iloc[[1,2],[1,2]]
        #    1    2
        # 1  1.0  1.0
        # 2  1.0  1.0
        #
        # Thus, we processing the index along two axis of DataFrame separately.
        shape0 = tuple(calc_shape((df.shape[0],), (self.indexes[0],)))
        shape1 = tuple(calc_shape((df.shape[1],), (self.indexes[1],)))

        inputs = [df] + [
            index for index in self.indexes if isinstance(index, ENTITY_TYPE)
        ]
        # NB: pandas only compresses the result to series when index on one of axis is integral
        if isinstance(self.indexes[1], Integral):
            shape = shape0
            dtype = df.dtypes.iloc[self.indexes[1]]
            index_value = indexing_index_value(df.index_value, self.indexes[0])
            if isinstance(self.indexes[0], Integral):
                # scalar
                return self.new_scalar(inputs, dtype=dtype)
            else:
                return self.new_series(
                    inputs,
                    shape=shape,
                    dtype=dtype,
                    index_value=index_value,
                    name=df.dtypes.index[self.indexes[1]],
                )
        elif isinstance(self.indexes[0], Integral):
            shape = shape1
            dtype = find_common_type(list(df.dtypes.iloc[self.indexes[1]].values))
            index_value = indexing_index_value(df.columns_value, self.indexes[1])
            return self.new_series(
                inputs, shape=shape, dtype=dtype, index_value=index_value
            )
        else:
            return self.new_dataframe(
                inputs,
                shape=shape0 + shape1,
                dtypes=df.dtypes.iloc[self.indexes[1]],
                index_value=indexing_index_value(df.index_value, self.indexes[0]),
                columns_value=indexing_index_value(
                    df.columns_value, self.indexes[1], store_data=True
                ),
            )


class SeriesIlocGetItem(DataFrameOperator, HeadTailOptimizedOperatorMixin):
    _op_module_ = "series"
    _op_type_ = opcodes.DATAFRAME_ILOC_GETITEM

    _input = KeyField("input")
    indexes = ListField("indexes", default=None)

    def __init__(self, gpu=None, sparse=False, output_types=None, **kw):
        super().__init__(gpu=gpu, sparse=sparse, _output_types=output_types, **kw)
        if not self.output_types:
            self.output_types = [OutputType.series]

    @property
    def input(self):
        return self._input

    @classmethod
    def _set_inputs(cls, op: "SeriesIlocGetItem", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)

        inputs_iter = iter(op._inputs)
        op._input = next(inputs_iter)

        indexes = []
        for index in op.indexes:
            if isinstance(index, ENTITY_TYPE):
                indexes.append(next(inputs_iter))
            else:
                indexes.append(index)
        op.indexes = indexes

    def __call__(self, series):
        if isinstance(self.indexes[0], Integral):
            return self.new_scalar([series], dtype=series.dtype)
        else:
            shape = tuple(calc_shape(series.shape, self.indexes))
            index_value = indexing_index_value(series.index_value, self.indexes[0])
            inputs = [series] + [
                index for index in self.indexes if isinstance(index, ENTITY_TYPE)
            ]
            return self.new_series(
                inputs,
                shape=shape,
                dtype=series.dtype,
                index_value=index_value,
                name=series.name,
            )


class IndexIlocGetItem(DataFrameOperator, DataFrameOperatorMixin):
    _op_module_ = "index"
    _op_type_ = opcodes.DATAFRAME_ILOC_GETITEM

    _input = KeyField("input")
    indexes = ListField("indexes", default=None)

    def __init__(self, gpu=None, sparse=False, output_types=None, **kw):
        super().__init__(gpu=gpu, sparse=sparse, _output_types=output_types, **kw)
        if not self.output_types:
            self.output_types = [OutputType.index]

    @property
    def input(self):
        return self._input

    @classmethod
    def _set_inputs(cls, op: "IndexIlocGetItem", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)

        inputs_iter = iter(op._inputs)
        op._input = next(inputs_iter)

        indexes = []
        for index in op.indexes:
            if isinstance(index, ENTITY_TYPE):
                indexes.append(next(inputs_iter))
            else:
                indexes.append(index)
        op.indexes = indexes

    def __call__(self, idx):
        if isinstance(self.indexes[0], Integral):
            return self.new_scalar([idx], dtype=idx.dtype)
        else:
            shape = tuple(calc_shape(idx.shape, self.indexes))
            index_value = indexing_index_value(idx.index_value, self.indexes[0])
            inputs = [idx] + [
                index for index in self.indexes if isinstance(index, ENTITY_TYPE)
            ]
            return self.new_index(
                inputs,
                shape=shape,
                dtype=idx.dtype,
                index_value=index_value,
                name=idx.name,
            )


class DataFrameIlocSetItem(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_ILOC_SETITEM

    indexes = ListField("indexes", default=None)
    value = AnyField("value", default=None)

    def __init__(self, gpu=None, sparse=False, output_types=None, **kw):
        super().__init__(
            gpu=gpu,
            sparse=sparse,
            _output_types=output_types,
            **kw,
        )
        if not self.output_types:
            self.output_types = [OutputType.dataframe]

    def __call__(self, df):
        return self.new_dataframe(
            [df],
            shape=df.shape,
            dtypes=df.dtypes,
            index_value=df.index_value,
            columns_value=df.columns_value,
        )


class SeriesIlocSetItem(DataFrameOperator, DataFrameOperatorMixin):
    _op_module_ = "series"
    _op_type_ = opcodes.DATAFRAME_ILOC_SETITEM

    indexes = ListField("indexes", default=None)
    value = AnyField("value", default=None)

    def __init__(self, gpu=None, sparse=False, **kw):
        super().__init__(
            gpu=gpu,
            sparse=sparse,
            _output_types=[OutputType.series],
            **kw,
        )

    def __call__(self, series):
        return self.new_series(
            [series],
            shape=series.shape,
            dtype=series.dtype,
            index_value=series.index_value,
            name=series.name,
        )


def index_getitem(idx, indexes):
    op = IndexIlocGetItem(indexes=process_iloc_indexes(idx, indexes))
    return op(idx)


def index_setitem(_idx, *_):
    raise TypeError("Index does not support mutable operations")


def iloc(a):
    """
    Purely integer-location based indexing for selection by position.

    ``.iloc[]`` is primarily integer position based (from ``0`` to
    ``length-1`` of the axis), but may also be used with a boolean
    array.

    Allowed inputs are:

    - An integer, e.g. ``5``.
    - A list or array of integers, e.g. ``[4, 3, 0]``.
    - A slice object with ints, e.g. ``1:7``.
    - A boolean array.
    - A ``callable`` function with one argument (the calling Series or
      DataFrame) and that returns valid output for indexing (one of the above).
      This is useful in method chains, when you don't have a reference to the
      calling object, but would like to base your selection on some value.

    ``.iloc`` will raise ``IndexError`` if a requested indexer is
    out-of-bounds, except *slice* indexers which allow out-of-bounds
    indexing (this conforms with python/numpy *slice* semantics).

    See more at :ref:`Selection by Position <indexing.integer>`.

    See Also
    --------
    DataFrame.iat : Fast integer location scalar accessor.
    DataFrame.loc : Purely label-location based indexer for selection by label.
    Series.iloc : Purely integer-location based indexing for
                   selection by position.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
    ...           {'a': 100, 'b': 200, 'c': 300, 'd': 400},
    ...           {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
    >>> df = md.DataFrame(mydict)
    >>> df.execute()
          a     b     c     d
    0     1     2     3     4
    1   100   200   300   400
    2  1000  2000  3000  4000

    **Indexing just the rows**

    With a scalar integer.

    >>> type(df.iloc[0]).execute()
    <class 'pandas.core.series.Series'>
    >>> df.iloc[0].execute()
    a    1
    b    2
    c    3
    d    4
    Name: 0, dtype: int64

    With a list of integers.

    >>> df.iloc[[0]].execute()
       a  b  c  d
    0  1  2  3  4
    >>> type(df.iloc[[0]]).execute()
    <class 'pandas.core.frame.DataFrame'>

    >>> df.iloc[[0, 1]].execute()
         a    b    c    d
    0    1    2    3    4
    1  100  200  300  400

    With a `slice` object.

    >>> df.iloc[:3].execute()
          a     b     c     d
    0     1     2     3     4
    1   100   200   300   400
    2  1000  2000  3000  4000

    With a boolean mask the same length as the index.

    >>> df.iloc[[True, False, True]].execute()
          a     b     c     d
    0     1     2     3     4
    2  1000  2000  3000  4000

    With a callable, useful in method chains. The `x` passed
    to the ``lambda`` is the DataFrame being sliced. This selects
    the rows whose index label even.

    >>> df.iloc[lambda x: x.index % 2 == 0].execute()
          a     b     c     d
    0     1     2     3     4
    2  1000  2000  3000  4000

    **Indexing both axes**

    You can mix the indexer types for the index and columns. Use ``:`` to
    select the entire axis.

    With scalar integers.

    >>> df.iloc[0, 1].execute()
    2

    With lists of integers.

    >>> df.iloc[[0, 2], [1, 3]].execute()
          b     d
    0     2     4
    2  2000  4000

    With `slice` objects.

    >>> df.iloc[1:3, 0:3].execute()
          a     b     c
    1   100   200   300
    2  1000  2000  3000

    With a boolean array whose length matches the columns.

    >>> df.iloc[:, [True, False, True, False]].execute()
          a     c
    0     1     3
    1   100   300
    2  1000  3000

    With a callable function that expects the Series or DataFrame.

    >>> df.iloc[:, lambda df: [0, 2]].execute()
          a     c
    0     1     3
    1   100   300
    2  1000  3000
    """
    return DataFrameIloc(a)


def head(a, n=5):
    """
    Return the first `n` rows.

    This function returns the first `n` rows for the object based
    on position. It is useful for quickly testing if your object
    has the right type of data in it.

    For negative values of `n`, this function returns all rows except
    the last `n` rows, equivalent to ``df[:-n]``.

    Parameters
    ----------
    n : int, default 5
        Number of rows to select.

    Returns
    -------
    same type as caller
        The first `n` rows of the caller object.

    See Also
    --------
    DataFrame.tail: Returns the last `n` rows.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'animal': ['alligator', 'bee', 'falcon', 'lion',
    ...                    'monkey', 'parrot', 'shark', 'whale', 'zebra']})
    >>> df.execute()
          animal
    0  alligator
    1        bee
    2     falcon
    3       lion
    4     monkey
    5     parrot
    6      shark
    7      whale
    8      zebra

    Viewing the first 5 lines

    >>> df.head().execute()
          animal
    0  alligator
    1        bee
    2     falcon
    3       lion
    4     monkey

    Viewing the first `n` lines (three in this case)

    >>> df.head(3).execute()
          animal
    0  alligator
    1        bee
    2     falcon

    For negative values of `n`

    >>> df.head(-3).execute()
          animal
    0  alligator
    1        bee
    2     falcon
    3       lion
    4     monkey
    5     parrot
    """
    return DataFrameIloc(a)[0:n]


def tail(a, n=5):
    """
    Return the last `n` rows.

    This function returns last `n` rows from the object based on
    position. It is useful for quickly verifying data, for example,
    after sorting or appending rows.

    For negative values of `n`, this function returns all rows except
    the first `n` rows, equivalent to ``df[n:]``.

    Parameters
    ----------
    n : int, default 5
        Number of rows to select.

    Returns
    -------
    type of caller
        The last `n` rows of the caller object.

    See Also
    --------
    DataFrame.head : The first `n` rows of the caller object.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'animal': ['alligator', 'bee', 'falcon', 'lion',
    ...                    'monkey', 'parrot', 'shark', 'whale', 'zebra']})
    >>> df.execute()
          animal
    0  alligator
    1        bee
    2     falcon
    3       lion
    4     monkey
    5     parrot
    6      shark
    7      whale
    8      zebra

    Viewing the last 5 lines

    >>> df.tail().execute()
       animal
    4  monkey
    5  parrot
    6   shark
    7   whale
    8   zebra

    Viewing the last `n` lines (three in this case)

    >>> df.tail(3).execute()
      animal
    6  shark
    7  whale
    8  zebra

    For negative values of `n`

    >>> df.tail(-3).execute()
       animal
    3    lion
    4  monkey
    5  parrot
    6   shark
    7   whale
    8   zebra
    """
    return DataFrameIloc(a)[-n:]
