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

import warnings
from typing import List

import numpy as np

from ... import opcodes
from ...core import Entity, EntityData, OutputType
from ...serialization.serializables import AnyField, StringField
from ..core import DATAFRAME_TYPE, SERIES_TYPE, IndexValue
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index, validate_axis


class DataFrameDrop(DataFrameOperatorMixin, DataFrameOperator):
    _op_type_ = opcodes.DATAFRAME_DROP

    index = AnyField("index", default=None)
    columns = AnyField("columns", default=None)
    level = AnyField("level", default=None)
    errors = StringField("errors", default="raise")

    def _filter_dtypes(self, dtypes, ignore_errors=False):
        if self.columns:
            return dtypes.drop(
                index=self.columns,
                level=self.level,
                errors="ignore" if ignore_errors else self.errors,
            )
        else:
            return dtypes

    @classmethod
    def _set_inputs(cls, op: "DataFrameDrop", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs[1:])
        if len(op._inputs) > 1:
            op.index = next(inputs_iter)

    def __call__(self, df_or_series):
        params = df_or_series.params.copy()
        shape_list = list(df_or_series.shape)

        if self.index is not None:
            if isinstance(df_or_series.index_value.value, IndexValue.RangeIndex):
                params["index_value"] = parse_index(
                    None, (df_or_series.key, df_or_series.index_value.key)
                )
            shape_list[0] = np.nan

        if isinstance(df_or_series, DATAFRAME_TYPE):
            new_dtypes = self._filter_dtypes(df_or_series.dtypes)
            params["columns_value"] = parse_index(new_dtypes.index, store_data=True)
            params["dtypes"] = new_dtypes
            shape_list[1] = len(new_dtypes)
            self.output_types = [OutputType.dataframe]
        elif isinstance(df_or_series, SERIES_TYPE):
            self.output_types = [OutputType.series]
        else:
            self.output_types = [OutputType.index]

        params["shape"] = tuple(shape_list)

        inputs = [df_or_series]
        if isinstance(self.index, Entity):
            inputs.append(self.index)
        return self.new_tileable(inputs, **params)


def _drop(
    df_or_series,
    labels=None,
    axis=0,
    index=None,
    columns=None,
    level=None,
    inplace=False,
    errors="raise",
):
    axis = validate_axis(axis, df_or_series)
    if labels is not None and (index is not None or columns is not None):
        raise ValueError("Cannot specify both 'labels' and 'index'/'columns'")
    if labels is not None:
        if axis == 0:
            index = labels
        else:
            columns = labels

    if index is not None and errors == "raise":
        warnings.warn("Errors will not raise for non-existing indices")
    if isinstance(columns, Entity):
        raise NotImplementedError("Columns cannot be MaxFrame objects")

    op = DataFrameDrop(index=index, columns=columns, level=level, errors=errors)
    df = op(df_or_series)
    if inplace:
        df_or_series.data = df.data
    else:
        return df


def df_drop(
    df,
    labels=None,
    axis=0,
    index=None,
    columns=None,
    level=None,
    inplace=False,
    errors="raise",
):
    """
    Drop specified labels from rows or columns.

    Remove rows or columns by specifying label names and corresponding
    axis, or by specifying directly index or column names. When using a
    multi-index, labels on different levels can be removed by specifying
    the level.

    Parameters
    ----------
    labels : single label or list-like
        Index or column labels to drop.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Whether to drop labels from the index (0 or 'index') or
        columns (1 or 'columns').
    index : single label or list-like
        Alternative to specifying axis (``labels, axis=0``
        is equivalent to ``index=labels``).
    columns : single label or list-like
        Alternative to specifying axis (``labels, axis=1``
        is equivalent to ``columns=labels``).
    level : int or level name, optional
        For MultiIndex, level from which the labels will be removed.
    inplace : bool, default False
        If True, do operation inplace and return None.
    errors : {'ignore', 'raise'}, default 'raise'
        If 'ignore', suppress error and only existing labels are
        dropped. Note that errors for missing indices will not raise.

    Returns
    -------
    DataFrame
        DataFrame without the removed index or column labels.

    Raises
    ------
    KeyError
        If any of the labels is not found in the selected axis.

    See Also
    --------
    DataFrame.loc : Label-location based indexer for selection by label.
    DataFrame.dropna : Return DataFrame with labels on given axis omitted
        where (all or any) data are missing.
    DataFrame.drop_duplicates : Return DataFrame with duplicate rows
        removed, optionally only considering certain columns.
    Series.drop : Return Series with specified index labels removed.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame(np.arange(12).reshape(3, 4),
    ...                   columns=['A', 'B', 'C', 'D'])
    >>> df.execute()
       A  B   C   D
    0  0  1   2   3
    1  4  5   6   7
    2  8  9  10  11

    Drop columns

    >>> df.drop(['B', 'C'], axis=1).execute()
       A   D
    0  0   3
    1  4   7
    2  8  11

    >>> df.drop(columns=['B', 'C']).execute()
       A   D
    0  0   3
    1  4   7
    2  8  11

    Drop a row by index

    >>> df.drop([0, 1]).execute()
       A  B   C   D
    2  8  9  10  11

    Drop columns and/or rows of MultiIndex DataFrame

    >>> midx = pd.MultiIndex(levels=[['lame', 'cow', 'falcon'],
    ...                              ['speed', 'weight', 'length']],
    ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],
    ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])
    >>> df = md.DataFrame(index=midx, columns=['big', 'small'],
    ...                   data=[[45, 30], [200, 100], [1.5, 1], [30, 20],
    ...                         [250, 150], [1.5, 0.8], [320, 250],
    ...                         [1, 0.8], [0.3, 0.2]])
    >>> df.execute()
                    big     small
    lame    speed   45.0    30.0
            weight  200.0   100.0
            length  1.5     1.0
    cow     speed   30.0    20.0
            weight  250.0   150.0
            length  1.5     0.8
    falcon  speed   320.0   250.0
            weight  1.0     0.8
            length  0.3     0.2

    >>> df.drop(index='cow', columns='small').execute()
                    big
    lame    speed   45.0
            weight  200.0
            length  1.5
    falcon  speed   320.0
            weight  1.0
            length  0.3

    >>> df.drop(index='length', level=1).execute()
                    big     small
    lame    speed   45.0    30.0
            weight  200.0   100.0
    cow     speed   30.0    20.0
            weight  250.0   150.0
    falcon  speed   320.0   250.0
            weight  1.0     0.8
    """
    return _drop(
        df,
        labels=labels,
        axis=axis,
        index=index,
        columns=columns,
        level=level,
        inplace=inplace,
        errors=errors,
    )


def df_pop(df, item):
    """
    Return item and drop from frame. Raise KeyError if not found.

    Parameters
    ----------
    item : str
        Label of column to be popped.

    Returns
    -------
    Series

    Examples
    --------
    >>> import numpy as np
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([('falcon', 'bird', 389.0),
    ...                    ('parrot', 'bird', 24.0),
    ...                    ('lion', 'mammal', 80.5),
    ...                    ('monkey', 'mammal', np.nan)],
    ...                   columns=('name', 'class', 'max_speed'))
    >>> df.execute()
         name   class  max_speed
    0  falcon    bird      389.0
    1  parrot    bird       24.0
    2    lion  mammal       80.5
    3  monkey  mammal        NaN

    >>> df.pop('class').execute()
    0      bird
    1      bird
    2    mammal
    3    mammal
    Name: class, dtype: object

    >>> df.execute()
         name  max_speed
    0  falcon      389.0
    1  parrot       24.0
    2    lion       80.5
    3  monkey        NaN
    """
    series = df.data[item]
    df_drop(df, item, axis=1, inplace=True)
    return series


def series_drop(
    series,
    labels=None,
    axis=0,
    index=None,
    columns=None,
    level=None,
    inplace=False,
    errors="raise",
):
    """
    Return Series with specified index labels removed.

    Remove elements of a Series based on specifying the index labels.
    When using a multi-index, labels on different levels can be removed
    by specifying the level.

    Parameters
    ----------
    labels : single label or list-like
        Index labels to drop.
    axis : 0, default 0
        Redundant for application on Series.
    index : single label or list-like
        Redundant for application on Series, but 'index' can be used instead
        of 'labels'.

        .. versionadded:: 0.21.0
    columns : single label or list-like
        No change is made to the Series; use 'index' or 'labels' instead.

        .. versionadded:: 0.21.0
    level : int or level name, optional
        For MultiIndex, level for which the labels will be removed.
    inplace : bool, default False
        If True, do operation inplace and return None.
    errors : {'ignore', 'raise'}, default 'raise'
        Note that this argument is kept only for compatibility, and errors
        will not raise even if ``errors=='raise'``.

    Returns
    -------
    Series
        Series with specified index labels removed.

    Raises
    ------
    KeyError
        If none of the labels are found in the index.

    See Also
    --------
    Series.reindex : Return only specified index labels of Series.
    Series.dropna : Return series without null values.
    Series.drop_duplicates : Return Series with duplicate values removed.
    DataFrame.drop : Drop specified labels from rows or columns.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import maxframe.dataframe as md
    >>> s = md.Series(data=np.arange(3), index=['A', 'B', 'C'])
    >>> s.execute()
    A  0
    B  1
    C  2
    dtype: int64

    Drop labels B en C

    >>> s.drop(labels=['B', 'C']).execute()
    A  0
    dtype: int64

    Drop 2nd level label in MultiIndex Series

    >>> midx = pd.MultiIndex(levels=[['lame', 'cow', 'falcon'],
    ...                              ['speed', 'weight', 'length']],
    ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],
    ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])
    >>> s = md.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
    ...               index=midx)
    >>> s.execute()
    lame    speed      45.0
            weight    200.0
            length      1.2
    cow     speed      30.0
            weight    250.0
            length      1.5
    falcon  speed     320.0
            weight      1.0
            length      0.3
    dtype: float64

    >>> s.drop(labels='weight', level=1).execute()
    lame    speed      45.0
            length      1.2
    cow     speed      30.0
            length      1.5
    falcon  speed     320.0
            length      0.3
    dtype: float64
    """
    return _drop(
        series,
        labels=labels,
        axis=axis,
        index=index,
        columns=columns,
        level=level,
        inplace=inplace,
        errors=errors,
    )


def series_pop(series, item):
    """
    Return item and drops from series. Raise KeyError if not found.

    Parameters
    ----------
    item : label
        Index of the element that needs to be removed.

    Returns
    -------
    Value that is popped from series.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> ser = md.Series([1,2,3])

    >>> ser.pop(0).execute()
    1

    >>> ser.execute()
    1    2
    2    3
    dtype: int64
    """
    scalar = series.data[item]
    series_drop(series, item, inplace=True)
    return scalar


def index_drop(index, labels, errors="raise"):
    """
    Make new Index with passed list of labels deleted.

    Parameters
    ----------
    labels : array-like
    errors : {'ignore', 'raise'}, default 'raise'
        Note that this argument is kept only for compatibility, and errors
        will not raise even if ``errors=='raise'``.

    Returns
    -------
    dropped : Index

    Raises
    ------
    KeyError
        If not all of the labels are found in the selected axis
    """
    return _drop(index, labels=labels, errors=errors)
