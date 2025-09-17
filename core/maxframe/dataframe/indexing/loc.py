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
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import find_common_type
from pandas.core.indexing import IndexingError

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData, OutputType
from ...serialization.serializables import AnyField, KeyField, ListField
from ...tensor.datasource import asarray
from ...tensor.utils import calc_sliced_size, filter_inputs
from ...utils import is_full_slice, lazy_import, pd_release_version
from ..core import DATAFRAME_TYPE, IndexValue
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index, validate_axis
from .iloc import DataFrameIlocSetItem

cudf = lazy_import("cudf")
with_slice_locs_kind = pd_release_version < (1, 4, 0)


def process_loc_indexes(inp, indexes, fetch_index: bool = True, axis=None):
    ndim = inp.ndim

    if not isinstance(indexes, tuple):
        indexes = (indexes,)
    if axis is not None and inp.axes[axis].nlevels > 1:
        indexes = (indexes,)
    if len(indexes) < ndim:
        if axis == 0 or axis is None:
            indexes += (slice(None),) * (ndim - len(indexes))
        else:
            indexes = (slice(None),) * (ndim - len(indexes)) + indexes
    if len(indexes) > ndim:
        raise IndexingError("Too many indexers")

    new_indexes = []
    for ax, index in enumerate(indexes):
        if isinstance(index, (list, np.ndarray, pd.Series, ENTITY_TYPE)):
            if not isinstance(index, ENTITY_TYPE):
                index = np.asarray(index)
            elif fetch_index:
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
        new_indexes.append(index)

    return new_indexes


class DataFrameLoc:
    def __init__(self, obj, axis=None):
        self._obj = obj
        self._axis = axis

    def _use_iloc(self, indexes):
        # for RangeIndex from 0, use iloc instead of loc
        index_value = self._obj.index_value.value
        if len(indexes) == 2:
            if not isinstance(indexes[1], slice):
                return False, None
            elif indexes[1] != slice(None):
                return False, None
        if not isinstance(index_value, IndexValue.RangeIndex):
            return False, None
        if index_value.slice.start != 0 and index_value.slice.start is not None:
            return False, None
        if not isinstance(indexes[0], (Integral, slice)):
            return False, None
        if isinstance(indexes[0], Integral):
            if indexes[0] < 0:
                return False, None
        else:
            index0 = indexes[0]
            for v in (index0.start, index0.stop, index0.step):
                if v is None:
                    continue
                if not isinstance(v, Integral):
                    return False, None
                if v < 0:
                    return False, None
            if index0.stop is not None:
                # adjust slice right bound
                return (
                    True,
                    [slice(index0.start, index0.stop + 1, index0.step)] + indexes[1:],
                )
        return True, None

    def __getitem__(self, indexes):
        indexes = process_loc_indexes(self._obj, indexes, axis=self._axis)

        use_iloc, new_indexes = self._use_iloc(indexes)
        if use_iloc:
            # use iloc instead
            return self._obj.iloc[tuple(new_indexes or indexes)]

        op = DataFrameLocGetItem(indexes=indexes)
        return op(self._obj)

    def __setitem__(self, indexes, value):
        if not np.isscalar(value):
            raise NotImplementedError("Only scalar value is supported to set by loc")
        if not isinstance(self._obj, DATAFRAME_TYPE):
            raise NotImplementedError("Only DataFrame is supported to set by loc")
        indexes = process_loc_indexes(
            self._obj, indexes, fetch_index=False, axis=self._axis
        )
        use_iloc, new_indexes = self._use_iloc(indexes)
        if use_iloc:
            op = DataFrameIlocSetItem(indexes=new_indexes, value=value)
            ret = op(self._obj)
            self._obj.data = ret.data
        else:
            other_indices = []
            indices_tileable = [
                idx
                for idx in indexes
                if isinstance(idx, ENTITY_TYPE) or other_indices.append(idx)
            ]
            op = DataFrameLocSetItem(indexes=other_indices, value=value)
            ret = op([self._obj] + indices_tileable)
            self._obj.data = ret.data

    def __call__(self, axis):
        axis = validate_axis(axis, self._obj)
        return DataFrameLoc(self._obj, axis)


class DataFrameLocSetItem(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_LOC_SETITEM

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

    def __call__(self, inputs):
        df = inputs[0]
        return self.new_dataframe(
            inputs,
            shape=df.shape,
            dtypes=df.dtypes,
            index_value=df.index_value,
            columns_value=df.columns_value,
        )


class DataFrameLocGetItem(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_LOC_GETITEM

    _input = KeyField("input")
    indexes = ListField("indexes", default=None)

    def __init__(self, gpu=None, sparse=False, output_types=None, **kw):
        super().__init__(gpu=gpu, sparse=sparse, _output_types=output_types, **kw)

    @property
    def input(self):
        return self._input

    @property
    def can_index_miss(self):
        return False

    @classmethod
    def _set_inputs(cls, op: "DataFrameLocGetItem", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs)
        op._input = next(inputs_iter)
        indexes = []
        for index in op.indexes:
            if isinstance(index, ENTITY_TYPE):
                indexes.append(next(inputs_iter))
            else:
                indexes.append(index)
        op.indexes = list(indexes)

    @classmethod
    def _calc_slice_param(
        cls,
        input_index_value: IndexValue,
        pd_index: pd.Index,
        inp,
        index: slice,
        axis: int,
    ) -> Dict:
        param = dict()
        if is_full_slice(index):
            # full slice on this axis
            param["shape"] = inp.shape[axis]
            param["index_value"] = input_index_value
            if axis == 1:
                param["dtypes"] = inp.dtypes
        elif input_index_value.has_value():
            kw = {}
            if with_slice_locs_kind:
                kw["kind"] = "loc"
            start, end = pd_index.slice_locs(index.start, index.stop, index.step, **kw)
            slc = slice(start, end, index.step)
            size = calc_sliced_size(inp.shape[axis], slc)
            param["shape"] = size
            out_index = pd_index[slc]
            param["index_value"] = parse_index(out_index, store_data=axis == 1)
            if axis == 1:
                param["dtypes"] = inp.dtypes[slc]
        else:
            assert axis == 0
            if index.start is None and index.stop is None:
                param["shape"] = calc_sliced_size(inp.shape[axis], index)
            else:
                param["shape"] = np.nan
            param["index_value"] = parse_index(pd_index, inp, index)

        return param

    @classmethod
    def _calc_bool_index_param(
        cls, input_index_value: IndexValue, pd_index: pd.Index, inp, index, axis: int
    ) -> Dict:
        param = dict()
        if input_index_value.has_value():
            if isinstance(index, np.ndarray):
                filtered_index = pd_index[index]
                param["shape"] = len(filtered_index)
                param["index_value"] = parse_index(filtered_index, store_data=axis == 1)
                if axis == 1:
                    param["dtypes"] = inp.dtypes[index]
            else:
                # tensor, cannot be indexer on axis 1
                assert axis == 0
                param["shape"] = np.nan
                param["index_value"] = parse_index(
                    pd.Index([], dtype=pd_index.dtype), inp, index, store_data=False
                )
        else:
            assert axis == 0
            if isinstance(index, np.ndarray):
                param["shape"] = int(index.sum())
            else:
                param["shape"] = np.nan
            param["index_value"] = parse_index(pd_index, inp, index, store_data=False)

        return param

    @classmethod
    def _calc_fancy_index_param(
        cls, input_index_value: IndexValue, pd_index: pd.Index, inp, index, axis: int
    ) -> Dict:
        param = dict()
        if input_index_value.has_value():
            if isinstance(index, np.ndarray):
                if not pd_index.is_unique:
                    assert axis == 1
                    # as there's no direct method in pandas to handle fancy indexes
                    # we creates a empty
                    new_dtypes = inp.dtypes.loc[index]
                    param["shape"] = len(new_dtypes)
                    param["index_value"] = parse_index(
                        new_dtypes.index, store_data=True
                    )
                    param["dtypes"] = new_dtypes
                else:
                    for it in index:
                        if it not in pd_index:
                            axis_name = "index" if axis == 0 else "columns"
                            raise KeyError(
                                f"Label [{it}] not found in the [{axis_name}]"
                            )
                    param["shape"] = len(index)
                    param["index_value"] = parse_index(pd.Index(index), store_data=True)
                    if axis == 1:
                        param["dtypes"] = inp.dtypes[index]
            else:
                assert axis == 0
                param["shape"] = index.shape[0]
                param["index_value"] = parse_index(
                    pd.Index([], dtype=pd_index.dtype), inp, index
                )
        else:
            assert axis == 0
            param["shape"] = index.shape[0]
            param["index_value"] = parse_index(pd_index, inp, index)

        return param

    @classmethod
    def _calc_param(cls, inp, axis: int, index) -> Dict:
        input_index_value = inp.index_value if axis == 0 else inp.columns_value
        pd_index = input_index_value.to_pandas()

        if isinstance(index, slice):
            return cls._calc_slice_param(input_index_value, pd_index, inp, index, axis)
        elif hasattr(index, "dtype") and index.ndim == 1:
            if index.dtype == np.bool_:
                # bool indexing
                return cls._calc_bool_index_param(
                    input_index_value, pd_index, inp, index, axis
                )
            else:
                # fancy indexing
                return cls._calc_fancy_index_param(
                    input_index_value, pd_index, inp, index, axis
                )
        else:
            param = dict()
            if input_index_value.has_value():
                loc = pd_index.get_loc(index)
                if isinstance(loc, (slice, np.ndarray)):
                    assert axis == 1
                    new_dtypes = inp.dtypes[loc]
                    param["shape"] = len(new_dtypes)
                    param["index_value"] = parse_index(
                        new_dtypes.index, store_data=True
                    )
                    param["dtypes"] = new_dtypes
                else:
                    # append None to indicate returning Series
                    param["shape"] = None
            elif isinstance(index, tuple):
                has_ranges = any(
                    isinstance(i, (slice, np.ndarray))
                    or (hasattr(i, "dtype") and index.ndim == 1)
                    for i in index
                )
                if has_ranges:
                    param["shape"] = np.nan
                    param["index_value"] = parse_index(pd_index, inp, index)
                else:
                    param["shape"] = None
            else:
                param["shape"] = None
            return param

    def __call__(self, inp):
        inputs = [inp] + filter_inputs(self.indexes)

        shape = []
        sizes = []
        index_value = columns_value = dtypes = None
        for ax, index in enumerate(self.indexes):
            param = self._calc_param(inp, ax, index)

            size = param.get("shape")
            sizes.append(size)
            if size is not None:
                shape.append(size)

            if ax == 0:
                index_value = param.get("index_value")
            else:
                columns_value = param.get("index_value")
                dtypes = param.get("dtypes")

        shape = tuple(shape)
        if len(shape) == 0:
            # scalar
            if isinstance(inp, DATAFRAME_TYPE):
                dtype = inp.dtypes[self.indexes[1]]
            else:
                dtype = inp.dtype
            return self.new_scalar(inputs, dtype=dtype)
        elif len(shape) == 1:
            # series
            if isinstance(inp, DATAFRAME_TYPE):
                if sizes[0] is None:
                    # label on axis 0
                    dtype = find_common_type(list(dtypes))
                    return self.new_series(
                        inputs,
                        shape=shape,
                        dtype=dtype,
                        index_value=columns_value,
                        name=self.indexes[0],
                    )
                else:
                    # label on axis 1
                    dtype = inp.dtypes[self.indexes[1]]
                    return self.new_series(
                        inputs,
                        shape=shape,
                        dtype=dtype,
                        index_value=index_value,
                        name=self.indexes[1],
                    )
            else:
                return self.new_series(
                    inputs,
                    shape=shape,
                    dtype=inp.dtype,
                    index_value=index_value,
                    name=inp.name,
                )
        else:
            # dataframe
            return self.new_dataframe(
                inputs,
                shape=shape,
                dtypes=dtypes,
                index_value=index_value,
                columns_value=columns_value,
            )


def loc(a):
    """
    Access a group of rows and columns by label(s) or a boolean array.

    ``.loc[]`` is primarily label based, but may also be used with a
    boolean array.

    Allowed inputs are:

    - A single label, e.g. ``5`` or ``'a'``, (note that ``5`` is
      interpreted as a *label* of the index, and **never** as an
      integer position along the index).
    - A list or array of labels, e.g. ``['a', 'b', 'c']``.
    - A slice object with labels, e.g. ``'a':'f'``.

      .. warning:: Note that contrary to usual python slices, **both** the
          start and the stop are included

    - A boolean array of the same length as the axis being sliced,
      e.g. ``[True, False, True]``.
    - An alignable boolean Series. The index of the key will be aligned before
      masking.
    - An alignable Index. The Index of the returned selection will be the input.
    - A ``callable`` function with one argument (the calling Series or
      DataFrame) and that returns valid output for indexing (one of the above)

    See more at :ref:`Selection by Label <indexing.label>`.

    Raises
    ------
    KeyError
        If any items are not found.
    IndexingError
        If an indexed key is passed and its index is unalignable to the frame index.

    See Also
    --------
    DataFrame.at : Access a single value for a row/column label pair.
    DataFrame.iloc : Access group of rows and columns by integer position(s).
    DataFrame.xs : Returns a cross-section (row(s) or column(s)) from the
        Series/DataFrame.
    Series.loc : Access group of values using labels.

    Examples
    --------
    **Getting values**

    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([[1, 2], [4, 5], [7, 8]],
    ...      index=['cobra', 'viper', 'sidewinder'],
    ...      columns=['max_speed', 'shield'])
    >>> df.execute()
                max_speed  shield
    cobra               1       2
    viper               4       5
    sidewinder          7       8

    Single label. Note this returns the row as a Series.

    >>> df.loc['viper'].execute()
    max_speed    4
    shield       5
    Name: viper, dtype: int64

    List of labels. Note using ``[[]]`` returns a DataFrame.

    >>> df.loc[['viper', 'sidewinder']].execute()
                max_speed  shield
    viper               4       5
    sidewinder          7       8

    Single label for row and column

    >>> df.loc['cobra', 'shield'].execute()
    2

    Slice with labels for row and single label for column. As mentioned
    above, note that both the start and stop of the slice are included.

    >>> df.loc['cobra':'viper', 'max_speed'].execute()
    cobra    1
    viper    4
    Name: max_speed, dtype: int64

    Boolean list with the same length as the row axis

    >>> df.loc[[False, False, True]].execute()
                max_speed  shield
    sidewinder          7       8

    Alignable boolean Series:

    >>> df.loc[md.Series([False, True, False],
    ...        index=['viper', 'sidewinder', 'cobra'])].execute()
                max_speed  shield
    sidewinder          7       8

    Index (same behavior as ``df.reindex``)

    >>> df.loc[md.Index(["cobra", "viper"], name="foo")].execute()
           max_speed  shield
    foo
    cobra          1       2
    viper          4       5

    Conditional that returns a boolean Series

    >>> df.loc[df['shield'] > 6].execute()
                max_speed  shield
    sidewinder          7       8

    Conditional that returns a boolean Series with column labels specified

    >>> df.loc[df['shield'] > 6, ['max_speed']].execute()
                max_speed
    sidewinder          7

    Callable that returns a boolean Series

    >>> df.loc[lambda df: df['shield'] == 8].execute()
                max_speed  shield
    sidewinder          7       8

    **Setting values**

    Set value for all items matching the list of labels

    >>> df.loc[['viper', 'sidewinder'], ['shield']] = 50
    >>> df.execute()
                max_speed  shield
    cobra               1       2
    viper               4      50
    sidewinder          7      50

    Set value for an entire row

    >>> df.loc['cobra'] = 10
    >>> df.execute()
                max_speed  shield
    cobra              10      10
    viper               4      50
    sidewinder          7      50

    Set value for an entire column

    >>> df.loc[:, 'max_speed'] = 30
    >>> df.execute()
                max_speed  shield
    cobra              30      10
    viper              30      50
    sidewinder         30      50

    Set value for rows matching callable condition

    >>> df.loc[df['shield'] > 35] = 0
    >>> df.execute()
                max_speed  shield
    cobra              30      10
    viper               0       0
    sidewinder          0       0

    **Getting values on a DataFrame with an index that has integer labels**

    Another example using integers for the index

    >>> df = md.DataFrame([[1, 2], [4, 5], [7, 8]],
    ...      index=[7, 8, 9], columns=['max_speed', 'shield'])
    >>> df.execute()
       max_speed  shield
    7          1       2
    8          4       5
    9          7       8

    Slice with integer labels for rows. As mentioned above, note that both
    the start and stop of the slice are included.

    >>> df.loc[7:9].execute()
       max_speed  shield
    7          1       2
    8          4       5
    9          7       8

    **Getting values with a MultiIndex**

    A number of examples using a DataFrame with a MultiIndex

    >>> tuples = [
    ...    ('cobra', 'mark i'), ('cobra', 'mark ii'),
    ...    ('sidewinder', 'mark i'), ('sidewinder', 'mark ii'),
    ...    ('viper', 'mark ii'), ('viper', 'mark iii')
    ... ]
    >>> index = md.MultiIndex.from_tuples(tuples)
    >>> values = [[12, 2], [0, 4], [10, 20],
    ...         [1, 4], [7, 1], [16, 36]]
    >>> df = md.DataFrame(values, columns=['max_speed', 'shield'], index=index)
    >>> df.execute()
                         max_speed  shield
    cobra      mark i           12       2
               mark ii           0       4
    sidewinder mark i           10      20
               mark ii           1       4
    viper      mark ii           7       1
               mark iii         16      36

    Single label. Note this returns a DataFrame with a single index.

    >>> df.loc['cobra'].execute()
             max_speed  shield
    mark i          12       2
    mark ii          0       4

    Single index tuple. Note this returns a Series.

    >>> df.loc[('cobra', 'mark ii')].execute()
    max_speed    0
    shield       4
    Name: (cobra, mark ii), dtype: int64

    Single label for row and column. Similar to passing in a tuple, this
    returns a Series.

    >>> df.loc['cobra', 'mark i'].execute()
    max_speed    12
    shield        2
    Name: (cobra, mark i), dtype: int64

    Single tuple. Note using ``[[]]`` returns a DataFrame.

    >>> df.loc[[('cobra', 'mark ii')]].execute()
                   max_speed  shield
    cobra mark ii          0       4

    Single tuple for the index with a single label for the column

    >>> df.loc[('cobra', 'mark i'), 'shield'].execute()
    2

    Slice from index tuple to single label

    >>> df.loc[('cobra', 'mark i'):'viper'].execute()
                         max_speed  shield
    cobra      mark i           12       2
               mark ii           0       4
    sidewinder mark i           10      20
               mark ii           1       4
    viper      mark ii           7       1
               mark iii         16      36

    Slice from index tuple to index tuple

    >>> df.loc[('cobra', 'mark i'):('viper', 'mark ii')].execute()
                        max_speed  shield
    cobra      mark i          12       2
               mark ii          0       4
    sidewinder mark i          10      20
               mark ii          1       4
    viper      mark ii          7       1
    """
    return DataFrameLoc(a)
