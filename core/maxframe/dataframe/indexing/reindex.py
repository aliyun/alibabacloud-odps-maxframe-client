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

from ...core.operator import OperatorStage

try:
    import scipy.sparse as sps
except ImportError:  # pragma: no cover
    sps = None

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData
from ...serialization.serializables import (
    AnyField,
    BoolField,
    Int64Field,
    KeyField,
    StringField,
)
from ...tensor import tensor as astensor
from ...utils import is_full_slice, lazy_import, pd_release_version
from ..core import INDEX_TYPE
from ..core import Index as DataFrameIndexType
from ..initializer import Index as asindex
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index, validate_axis_style_args

cudf = lazy_import("cudf")

# under pandas<1.1, SparseArray ignores zeros on creation
_pd_sparse_miss_zero = pd_release_version[:2] < (1, 1)


class DataFrameReindex(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.REINDEX

    _input = KeyField("input")
    index = AnyField("index", default=None)
    index_freq = AnyField("index_freq", default=None)
    columns = AnyField("columns", default=None)
    method = StringField("method", default=None)
    level = AnyField("level", default=None)
    fill_value = AnyField("fill_value", default=None)
    limit = Int64Field("limit", default=None)
    tolerance = Int64Field("tolerance", default=None)
    enable_sparse = BoolField("enable_sparse", default=None)

    @property
    def input(self):
        return self._input

    @property
    def _indexes(self):
        # used for index_lib
        indexes = []
        names = ("index", "columns")
        for ax in range(self.input.ndim):
            index = names[ax]
            val = getattr(self, index)
            if val is not None:
                indexes.append(val)
            else:
                indexes.append(slice(None))
        return indexes

    @_indexes.setter
    def _indexes(self, new_indexes):
        for index_field, new_index in zip(["index", "columns"], new_indexes):
            setattr(self, index_field, new_index)

    @property
    def indexes(self):
        return self._indexes

    @property
    def can_index_miss(self):
        return True

    @classmethod
    def _set_inputs(cls, op: "DataFrameReindex", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if getattr(op, "indexes", None):
            op.index, op.columns = [
                None if is_full_slice(idx) else idx for idx in list(op.indexes) + [None]
            ][:2]
        inputs_iter = iter(inputs)
        op._input = next(inputs_iter)
        if op.index is not None and isinstance(op.index, ENTITY_TYPE):
            op.index = next(inputs_iter)
        if op.fill_value is not None:
            if op.stage == OperatorStage.agg:
                op.fill_value = None
            elif isinstance(op.fill_value, ENTITY_TYPE):
                op.fill_value = next(inputs_iter)

    def __call__(self, df_or_series):
        inputs = [df_or_series]
        shape = list(df_or_series.shape)
        index_value = df_or_series.index_value
        columns_value = dtypes = None
        if df_or_series.ndim == 2:
            columns_value = df_or_series.columns_value
            dtypes = df_or_series.dtypes

        if self.index is not None:
            shape[0] = self.index.shape[0]
            index_value = asindex(self.index).index_value
            self.index = astensor(self.index)
            if isinstance(self.index, ENTITY_TYPE):
                inputs.append(self.index)
        if self.columns is not None:
            shape[1] = self.columns.shape[0]
            dtypes = df_or_series.dtypes.reindex(index=self.columns).fillna(
                np.dtype(np.float64)
            )
            columns_value = parse_index(dtypes.index, store_data=True)
        if self.fill_value is not None and isinstance(self.fill_value, ENTITY_TYPE):
            inputs.append(self.fill_value)

        if df_or_series.ndim == 1:
            return self.new_series(
                inputs,
                shape=tuple(shape),
                dtype=df_or_series.dtype,
                index_value=index_value,
                name=df_or_series.name,
            )
        else:
            return self.new_dataframe(
                inputs,
                shape=tuple(shape),
                dtypes=dtypes,
                index_value=index_value,
                columns_value=columns_value,
            )


def reindex(
    df_or_series,
    labels=None,
    *,
    index=None,
    columns=None,
    axis=None,
    method=None,
    copy=None,
    level=None,
    fill_value=None,
    limit=None,
    tolerance=None,
    enable_sparse=False
):
    """
    Conform Series/DataFrame to new index with optional filling logic.

    Places NA/NaN in locations having no value in the previous index. A new object
    is produced unless the new index is equivalent to the current one and
    ``copy=False``.

    Parameters
    ----------
    labels : array-like, optional
        New labels / index to conform the axis specified by 'axis' to.
    index, columns : array-like, optional
        New labels / index to conform to, should be specified using
        keywords. Preferably an Index object to avoid duplicating data.
    axis : int or str, optional
        Axis to target. Can be either the axis name ('index', 'columns')
        or number (0, 1).
    method : {None, 'backfill'/'bfill', 'pad'/'ffill', 'nearest'}
        Method to use for filling holes in reindexed DataFrame.
        Please note: this is only applicable to DataFrames/Series with a
        monotonically increasing/decreasing index.

        * None (default): don't fill gaps
        * pad / ffill: Propagate last valid observation forward to next
          valid.
        * backfill / bfill: Use next valid observation to fill gap.
        * nearest: Use nearest valid observations to fill gap.

    copy : bool, default True
        Return a new object, even if the passed indexes are the same.
    level : int or name
        Broadcast across a level, matching Index values on the
        passed MultiIndex level.
    fill_value : scalar, default np.NaN
        Value to use for missing values. Defaults to NaN, but can be any
        "compatible" value.
    limit : int, default None
        Maximum number of consecutive elements to forward or backward fill.
    tolerance : optional
        Maximum distance between original and new labels for inexact
        matches. The values of the index at the matching locations most
        satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

        Tolerance may be a scalar value, which applies the same tolerance
        to all values, or list-like, which applies variable tolerance per
        element. List-like includes list, tuple, array, Series, and must be
        the same size as the index and its dtype must exactly match the
        index's type.

    Returns
    -------
    Series/DataFrame with changed index.

    See Also
    --------
    DataFrame.set_index : Set row labels.
    DataFrame.reset_index : Remove row labels or move them to new columns.
    DataFrame.reindex_like : Change to same indices as other DataFrame.

    Examples
    --------

    ``DataFrame.reindex`` supports two calling conventions

    * ``(index=index_labels, columns=column_labels, ...)``
    * ``(labels, axis={'index', 'columns'}, ...)``

    We *highly* recommend using keyword arguments to clarify your
    intent.

    Create a dataframe with some fictional data.

    >>> import maxframe.dataframe as md
    >>> index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']
    >>> df = md.DataFrame({'http_status': [200, 200, 404, 404, 301],
    ...                   'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]},
    ...                   index=index)
    >>> df.execute()
               http_status  response_time
    Firefox            200           0.04
    Chrome             200           0.02
    Safari             404           0.07
    IE10               404           0.08
    Konqueror          301           1.00

    Create a new index and reindex the dataframe. By default
    values in the new index that do not have corresponding
    records in the dataframe are assigned ``NaN``.

    >>> new_index = ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10',
    ...              'Chrome']
    >>> df.reindex(new_index).execute()
                   http_status  response_time
    Safari               404.0           0.07
    Iceweasel              NaN            NaN
    Comodo Dragon          NaN            NaN
    IE10                 404.0           0.08
    Chrome               200.0           0.02

    We can fill in the missing values by passing a value to
    the keyword ``fill_value``. Because the index is not monotonically
    increasing or decreasing, we cannot use arguments to the keyword
    ``method`` to fill the ``NaN`` values.

    >>> df.reindex(new_index, fill_value=0).execute()
                   http_status  response_time
    Safari                 404           0.07
    Iceweasel                0           0.00
    Comodo Dragon            0           0.00
    IE10                   404           0.08
    Chrome                 200           0.02

    >>> df.reindex(new_index, fill_value='missing').execute()
                  http_status response_time
    Safari                404          0.07
    Iceweasel         missing       missing
    Comodo Dragon     missing       missing
    IE10                  404          0.08
    Chrome                200          0.02

    We can also reindex the columns.

    >>> df.reindex(columns=['http_status', 'user_agent']).execute()
               http_status  user_agent
    Firefox            200         NaN
    Chrome             200         NaN
    Safari             404         NaN
    IE10               404         NaN
    Konqueror          301         NaN

    Or we can use "axis-style" keyword arguments

    >>> df.reindex(['http_status', 'user_agent'], axis="columns").execute()
               http_status  user_agent
    Firefox            200         NaN
    Chrome             200         NaN
    Safari             404         NaN
    IE10               404         NaN
    Konqueror          301         NaN

    To further illustrate the filling functionality in
    ``reindex``, we will create a dataframe with a
    monotonically increasing index (for example, a sequence
    of dates).

    >>> date_index = md.date_range('1/1/2010', periods=6, freq='D')
    >>> df2 = md.DataFrame({"prices": [100, 101, np.nan, 100, 89, 88]},
    ...                    index=date_index)
    >>> df2.execute()
                prices
    2010-01-01   100.0
    2010-01-02   101.0
    2010-01-03     NaN
    2010-01-04   100.0
    2010-01-05    89.0
    2010-01-06    88.0

    Suppose we decide to expand the dataframe to cover a wider
    date range.

    >>> date_index2 = md.date_range('12/29/2009', periods=10, freq='D')
    >>> df2.reindex(date_index2).execute()
                prices
    2009-12-29     NaN
    2009-12-30     NaN
    2009-12-31     NaN
    2010-01-01   100.0
    2010-01-02   101.0
    2010-01-03     NaN
    2010-01-04   100.0
    2010-01-05    89.0
    2010-01-06    88.0
    2010-01-07     NaN

    The index entries that did not have a value in the original data frame
    (for example, '2009-12-29') are by default filled with ``NaN``.
    If desired, we can fill in the missing values using one of several
    options.

    For example, to back-propagate the last valid value to fill the ``NaN``
    values, pass ``bfill`` as an argument to the ``method`` keyword.

    >>> df2.reindex(date_index2, method='bfill').execute()
                prices
    2009-12-29   100.0
    2009-12-30   100.0
    2009-12-31   100.0
    2010-01-01   100.0
    2010-01-02   101.0
    2010-01-03     NaN
    2010-01-04   100.0
    2010-01-05    89.0
    2010-01-06    88.0
    2010-01-07     NaN

    Please note that the ``NaN`` value present in the original dataframe
    (at index value 2010-01-03) will not be filled by any of the
    value propagation schemes. This is because filling while reindexing
    does not look at dataframe values, but only compares the original and
    desired indexes. If you do want to fill in the ``NaN`` values present
    in the original dataframe, use the ``fillna()`` method.

    See the :ref:`user guide <basics.reindexing>` for more.
    """
    axes_kwargs = dict(index=index, columns=columns, axis=axis)
    axes = validate_axis_style_args(
        df_or_series,
        (labels,) if labels is not None else (),
        {k: v for k, v in axes_kwargs.items() if v is not None},
        "labels",
        "reindex",
    )

    if tolerance is not None:  # pragma: no cover
        raise NotImplementedError("`tolerance` is not supported yet")

    if method == "nearest":  # pragma: no cover
        raise NotImplementedError("method=nearest is not supported yet")

    index = axes.get("index")
    index_freq = None
    if isinstance(index, ENTITY_TYPE):
        if isinstance(index, DataFrameIndexType):
            index_freq = getattr(index.index_value.value, "freq", None)
        if not isinstance(index, INDEX_TYPE):
            index = astensor(index)
    elif index is not None:
        index = np.asarray(index)
        index_freq = getattr(index, "freq", None)

    columns = axes.get("columns")
    if isinstance(columns, ENTITY_TYPE):  # pragma: no cover
        try:
            columns = columns.fetch()
        except ValueError:
            raise NotImplementedError(
                "`columns` need to be executed first if it's a MaxFrame object"
            )
    elif columns is not None:
        columns = np.asarray(columns)

    if isinstance(fill_value, ENTITY_TYPE) and getattr(fill_value, "ndim", 0) != 0:
        raise ValueError("fill_value must be a scalar")

    op = DataFrameReindex(
        index=index,
        index_freq=index_freq,
        columns=columns,
        method=method,
        level=level,
        fill_value=fill_value,
        limit=limit,
        enable_sparse=enable_sparse,
    )
    ret = op(df_or_series)

    if copy:
        return ret.copy()
    return ret


def reindex_like(
    df_or_series, other, method=None, copy=True, limit=None, tolerance=None
):
    """
    Return an object with matching indices as other object.

    Conform the object to the same index on all axes. Optional
    filling logic, placing NaN in locations having no value
    in the previous index. A new object is produced unless the
    new index is equivalent to the current one and copy=False.

    Parameters
    ----------
    other : Object of the same data type
        Its row and column indices are used to define the new indices
        of this object.
    method : {None, 'backfill'/'bfill', 'pad'/'ffill', 'nearest'}
        Method to use for filling holes in reindexed DataFrame.
        Please note: this is only applicable to DataFrames/Series with a
        monotonically increasing/decreasing index.

        * None (default): don't fill gaps
        * pad / ffill: propagate last valid observation forward to next
          valid
        * backfill / bfill: use next valid observation to fill gap
        * nearest: use nearest valid observations to fill gap.

    copy : bool, default True
        Return a new object, even if the passed indexes are the same.
    limit : int, default None
        Maximum number of consecutive labels to fill for inexact matches.
    tolerance : optional
        Maximum distance between original and new labels for inexact
        matches. The values of the index at the matching locations must
        satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

        Tolerance may be a scalar value, which applies the same tolerance
        to all values, or list-like, which applies variable tolerance per
        element. List-like includes list, tuple, array, Series, and must be
        the same size as the index and its dtype must exactly match the
        index's type.

    Returns
    -------
    Series or DataFrame
        Same type as caller, but with changed indices on each axis.

    See Also
    --------
    DataFrame.set_index : Set row labels.
    DataFrame.reset_index : Remove row labels or move them to new columns.
    DataFrame.reindex : Change to new indices or expand indices.

    Notes
    -----
    Same as calling
    ``.reindex(index=other.index, columns=other.columns,...)``.

    Examples
    --------
    >>> import pandas as pd
    >>> import maxframe.dataframe as md
    >>> df1 = md.DataFrame([[24.3, 75.7, 'high'],
    ...                     [31, 87.8, 'high'],
    ...                     [22, 71.6, 'medium'],
    ...                     [35, 95, 'medium']],
    ...                    columns=['temp_celsius', 'temp_fahrenheit',
    ...                             'windspeed'],
    ...                    index=md.date_range(start='2014-02-12',
    ...                                        end='2014-02-15', freq='D'))

    >>> df1.execute()
               temp_celsius temp_fahrenheit windspeed
    2014-02-12         24.3            75.7      high
    2014-02-13           31            87.8      high
    2014-02-14           22            71.6    medium
    2014-02-15           35              95    medium

    >>> df2 = md.DataFrame([[28, 'low'],
    ...                     [30, 'low'],
    ...                     [35.1, 'medium']],
    ...                    columns=['temp_celsius', 'windspeed'],
    ...                    index=pd.DatetimeIndex(['2014-02-12', '2014-02-13',
    ...                                            '2014-02-15']))

    >>> df2.execute()
                temp_celsius windspeed
    2014-02-12          28.0       low
    2014-02-13          30.0       low
    2014-02-15          35.1    medium

    >>> df2.reindex_like(df1).execute()
                temp_celsius  temp_fahrenheit windspeed
    2014-02-12          28.0              NaN       low
    2014-02-13          30.0              NaN       low
    2014-02-14           NaN              NaN       NaN
    2014-02-15          35.1              NaN    medium
    """
    cond = df_or_series.index_value.key == other.index_value.key
    if df_or_series.ndim == 2:
        cond &= df_or_series.columns_value.key == other.columns_value.key
    if cond and not copy:
        return df_or_series

    kw = {
        "index": other.index,
        "method": method,
        "limit": limit,
        "tolerance": tolerance,
    }
    if df_or_series.ndim == 2:
        kw["columns"] = other.dtypes.index
    return reindex(df_or_series, **kw)
