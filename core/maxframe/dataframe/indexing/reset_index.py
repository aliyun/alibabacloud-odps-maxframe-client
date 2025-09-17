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
import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import AnyField, BoolField
from ...utils import no_default, pd_release_version
from ..operators import DATAFRAME_TYPE, DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_empty_df, build_empty_series, parse_index

_reset_index_has_names = pd_release_version >= (1, 5)


class DataFrameResetIndex(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.RESET_INDEX

    level = AnyField("level", default=None)
    drop = BoolField("drop", default=False)
    name = AnyField("name", default=None)
    col_level = AnyField("col_level", default=None)
    col_fill = AnyField("col_fill", default=None)
    incremental_index = BoolField("incremental_index", default=False)
    names = AnyField("names", default=None)

    def __init__(self, output_types=None, **kwargs):
        super().__init__(_output_types=output_types, **kwargs)

    @classmethod
    def _get_out_index(cls, df, out_shape):
        if isinstance(df.index, pd.RangeIndex):
            range_value = -1 if np.isnan(out_shape[0]) else out_shape[0]
            index_value = parse_index(pd.RangeIndex(range_value))
        else:
            index_value = parse_index(df.index)
        return index_value

    def _call_series(self, a):
        if self.drop:
            range_value = -1 if np.isnan(a.shape[0]) else a.shape[0]
            index_value = parse_index(pd.RangeIndex(range_value))
            return self.new_series(
                [a], shape=a.shape, dtype=a.dtype, name=a.name, index_value=index_value
            )
        else:
            empty_series = build_empty_series(
                dtype=a.dtype, index=a.index_value.to_pandas()[:0], name=a.name
            )
            empty_df = empty_series.reset_index(level=self.level, name=self.name)
            shape = (a.shape[0], len(empty_df.dtypes))
            index_value = self._get_out_index(empty_df, shape)
            return self.new_dataframe(
                [a],
                shape=shape,
                index_value=index_value,
                columns_value=parse_index(empty_df.columns),
                dtypes=empty_df.dtypes,
            )

    def _call_dataframe(self, a):
        if self.drop:
            shape = a.shape
            columns_value = a.columns_value
            dtypes = a.dtypes
            range_value = -1 if np.isnan(a.shape[0]) else a.shape[0]
            index_value = parse_index(pd.RangeIndex(range_value))
        else:
            empty_df = build_empty_df(a.dtypes)
            empty_df.index = a.index_value.to_pandas()[:0]

            if self.names and _reset_index_has_names:
                empty_df = empty_df.reset_index(
                    level=self.level,
                    col_level=self.col_level,
                    col_fill=self.col_fill,
                    names=self.names,
                )
            else:
                empty_df = empty_df.reset_index(
                    level=self.level, col_level=self.col_level, col_fill=self.col_fill
                )
                if self.names:
                    names = (
                        [self.names] if not isinstance(self.names, list) else self.names
                    )
                    cols = list(empty_df.columns)
                    cols[: len(names)] = names
                    empty_df.columns = pd.Index(cols, name=empty_df.columns.name)

            shape = (a.shape[0], len(empty_df.columns))
            columns_value = parse_index(empty_df.columns, store_data=True)
            dtypes = empty_df.dtypes
            index_value = self._get_out_index(empty_df, shape)
        return self.new_dataframe(
            [a],
            shape=shape,
            columns_value=columns_value,
            index_value=index_value,
            dtypes=dtypes,
        )

    def __call__(self, a):
        if isinstance(a, DATAFRAME_TYPE):
            return self._call_dataframe(a)
        else:
            return self._call_series(a)


def df_reset_index(
    df,
    level=None,
    drop=False,
    inplace=False,
    col_level=0,
    col_fill="",
    names=None,
    incremental_index=False,
):
    """
    Reset the index, or a level of it.

    Reset the index of the DataFrame, and use the default one instead.
    If the DataFrame has a MultiIndex, this method can remove one or more
    levels.

    Parameters
    ----------
    level : int, str, tuple, or list, default None
        Only remove the given levels from the index. Removes all levels by
        default.
    drop : bool, default False
        Do not try to insert index into dataframe columns. This resets
        the index to the default integer index.
    inplace : bool, default False
        Modify the DataFrame in place (do not create a new object).
    col_level : int or str, default 0
        If the columns have multiple levels, determines which level the
        labels are inserted into. By default it is inserted into the first
        level.
    col_fill : object, default ''
        If the columns have multiple levels, determines how the other
        levels are named. If None then the index name is repeated.

    Returns
    -------
    DataFrame or None
        DataFrame with the new index or None if ``inplace=True``.

    See Also
    --------
    DataFrame.set_index : Opposite of reset_index.
    DataFrame.reindex : Change to new indices or expand indices.
    DataFrame.reindex_like : Change to same indices as other DataFrame.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([('bird', 389.0),
    ...                    ('bird', 24.0),
    ...                    ('mammal', 80.5),
    ...                    ('mammal', mt.nan)],
    ...                   index=['falcon', 'parrot', 'lion', 'monkey'],
    ...                   columns=('class', 'max_speed'))
    >>> df.execute()
             class  max_speed
    falcon    bird      389.0
    parrot    bird       24.0
    lion    mammal       80.5
    monkey  mammal        NaN

    When we reset the index, the old index is added as a column, and a
    new sequential index is used:

    >>> df.reset_index().execute()
        index   class  max_speed
    0  falcon    bird      389.0
    1  parrot    bird       24.0
    2    lion  mammal       80.5
    3  monkey  mammal        NaN

    We can use the `drop` parameter to avoid the old index being added as
    a column:

    >>> df.reset_index(drop=True).execute()
        class  max_speed
    0    bird      389.0
    1    bird       24.0
    2  mammal       80.5
    3  mammal        NaN

    You can also use `reset_index` with `MultiIndex`.

    >>> import pandas as pd
    >>> index = pd.MultiIndex.from_tuples([('bird', 'falcon'),
    ...                                    ('bird', 'parrot'),
    ...                                    ('mammal', 'lion'),
    ...                                    ('mammal', 'monkey')],
    ...                                   names=['class', 'name'])
    >>> columns = pd.MultiIndex.from_tuples([('speed', 'max'),
    ...                                      ('species', 'type')])
    >>> df = md.DataFrame([(389.0, 'fly'),
    ...                    ( 24.0, 'fly'),
    ...                    ( 80.5, 'run'),
    ...                    (mt.nan, 'jump')],
    ...                   index=index,
    ...                   columns=columns)
    >>> df.execute()
                   speed species
                     max    type
    class  name
    bird   falcon  389.0     fly
           parrot   24.0     fly
    mammal lion     80.5     run
           monkey    NaN    jump

    If the index has multiple levels, we can reset a subset of them:

    >>> df.reset_index(level='class').execute()
             class  speed species
                      max    type
    name
    falcon    bird  389.0     fly
    parrot    bird   24.0     fly
    lion    mammal   80.5     run
    monkey  mammal    NaN    jump

    If we are not dropping the index, by default, it is placed in the top
    level. We can place it in another level:

    >>> df.reset_index(level='class', col_level=1).execute()
                    speed species
             class    max    type
    name
    falcon    bird  389.0     fly
    parrot    bird   24.0     fly
    lion    mammal   80.5     run
    monkey  mammal    NaN    jump

    When the index is inserted under another level, we can specify under
    which one with the parameter `col_fill`:

    >>> df.reset_index(level='class', col_level=1, col_fill='species').execute()
                  species  speed species
                    class    max    type
    name
    falcon           bird  389.0     fly
    parrot           bird   24.0     fly
    lion           mammal   80.5     run
    monkey         mammal    NaN    jump

    If we specify a nonexistent level for `col_fill`, it is created:

    >>> df.reset_index(level='class', col_level=1, col_fill='genus').execute()
                    genus  speed species
                    class    max    type
    name
    falcon           bird  389.0     fly
    parrot           bird   24.0     fly
    lion           mammal   80.5     run
    monkey         mammal    NaN    jump
    """
    op = DataFrameResetIndex(
        level=level,
        drop=drop,
        col_level=col_level,
        col_fill=col_fill,
        names=names,
        incremental_index=incremental_index,
        output_types=[OutputType.dataframe],
    )
    ret = op(df)
    if not inplace:
        return ret
    else:
        df.data = ret.data


def series_reset_index(
    series,
    level=None,
    drop=False,
    name=no_default,
    inplace=False,
    incremental_index=False,
):
    """
    Generate a new DataFrame or Series with the index reset.

    This is useful when the index needs to be treated as a column, or
    when the index is meaningless and needs to be reset to the default
    before another operation.

    Parameters
    ----------
    level : int, str, tuple, or list, default optional
        For a Series with a MultiIndex, only remove the specified levels
        from the index. Removes all levels by default.
    drop : bool, default False
        Just reset the index, without inserting it as a column in
        the new DataFrame.
    name : object, optional
        The name to use for the column containing the original Series
        values. Uses ``self.name`` by default. This argument is ignored
        when `drop` is True.
    inplace : bool, default False
        Modify the Series in place (do not create a new object).

    Returns
    -------
    Series or DataFrame
        When `drop` is False (the default), a DataFrame is returned.
        The newly created columns will come first in the DataFrame,
        followed by the original Series values.
        When `drop` is True, a `Series` is returned.
        In either case, if ``inplace=True``, no value is returned.

    See Also
    --------
    DataFrame.reset_index: Analogous function for DataFrame.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> s = md.Series([1, 2, 3, 4], name='foo',
    ...               index=md.Index(['a', 'b', 'c', 'd'], name='idx'))

    Generate a DataFrame with default index.

    >>> s.reset_index().execute()
      idx  foo
    0   a    1
    1   b    2
    2   c    3
    3   d    4

    To specify the name of the new column use `name`.

    >>> s.reset_index(name='values').execute()
      idx  values
    0   a       1
    1   b       2
    2   c       3
    3   d       4

    To generate a new Series with the default set `drop` to True.

    >>> s.reset_index(drop=True).execute()
    0    1
    1    2
    2    3
    3    4
    Name: foo, dtype: int64

    To update the Series in place, without generating a new one
    set `inplace` to True. Note that it also requires ``drop=True``.

    >>> s.reset_index(inplace=True, drop=True)
    >>> s.execute()
    0    1
    1    2
    2    3
    3    4
    Name: foo, dtype: int64

    The `level` parameter is interesting for Series with a multi-level
    index.

    >>> import numpy as np
    >>> import pandas as pd
    >>> arrays = [np.array(['bar', 'bar', 'baz', 'baz']),
    ...           np.array(['one', 'two', 'one', 'two'])]
    >>> s2 = md.Series(
    ...     range(4), name='foo',
    ...     index=pd.MultiIndex.from_arrays(arrays,
    ...                                     names=['a', 'b']))

    To remove a specific level from the Index, use `level`.

    >>> s2.reset_index(level='a').execute()
           a  foo
    b
    one  bar    0
    two  bar    1
    one  baz    2
    two  baz    3

    If `level` is not set, all levels are removed from the Index.

    >>> s2.reset_index().execute()
         a    b  foo
    0  bar  one    0
    1  bar  two    1
    2  baz  one    2
    3  baz  two    3
    """
    if name is no_default:
        name = series.name if series.name is not None else 0

    op = DataFrameResetIndex(
        level=level,
        drop=drop,
        name=name,
        incremental_index=incremental_index,
        output_types=[OutputType.series if drop else OutputType.dataframe],
    )
    ret = op(series)
    if not inplace:
        return ret
    elif ret.ndim == 2:
        raise TypeError("Cannot reset_index inplace on a Series to create a DataFrame")
    else:
        series.data = ret.data
