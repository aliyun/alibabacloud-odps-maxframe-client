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
from ...serialization.serializables import AnyField, BoolField, Int32Field, StringField
from ...utils import no_default, pd_release_version
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index, validate_axis

_drop_na_enable_no_default = pd_release_version[:2] >= (1, 5)


class DataFrameDropNA(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DROP_NA

    axis = AnyField("axis", default=None)
    how = StringField("how", default=None)
    thresh = Int32Field("thresh", default=None)
    subset = AnyField("subset", default=None)

    # when True, dropna will be called on the input,
    # otherwise non-nan counts will be used
    drop_directly = BoolField("drop_directly", default=None)
    # size of subset, used when how == 'any'
    subset_size = Int32Field("subset_size", default=None)
    # if True, drop index
    ignore_index = BoolField("ignore_index", default=False)

    use_inf_as_na = BoolField("use_inf_as_na", default=None)

    def __init__(self, sparse=None, output_types=None, **kw):
        super().__init__(_output_types=output_types, sparse=sparse, **kw)

    def __call__(self, df):
        new_shape = list(df.shape)
        new_shape[0] = np.nan

        params = df.params.copy()
        params["index_value"] = parse_index(None, df.key, df.index_value.key)
        params["shape"] = tuple(new_shape)
        return self.new_tileable([df], **params)


def df_dropna(
    df,
    axis=0,
    how=no_default,
    thresh=no_default,
    subset=None,
    inplace=False,
    ignore_index=False,
):
    """
    Remove missing values.

    See the :ref:`User Guide <missing_data>` for more on which values are
    considered missing, and how to work with missing data.

    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Determine if rows or columns which contain missing values are
        removed.

        * 0, or 'index' : Drop rows which contain missing values.
        * 1, or 'columns' : Drop columns which contain missing value.

    how : {'any', 'all'}, default 'any'
        Determine if row or column is removed from DataFrame, when we have
        at least one NA or all NA.

        * 'any' : If any NA values are present, drop that row or column.
        * 'all' : If all values are NA, drop that row or column.

    thresh : int, optional
        Require that many non-NA values.
    subset : array-like, optional
        Labels along other axis to consider, e.g. if you are dropping rows
        these would be a list of columns to include.
    inplace : bool, default False
        If True, do operation inplace and return None.
    ignore_index : bool, default False
        If True, the resulting axis will be labeled 0, 1, …, n - 1.

    Returns
    -------
    DataFrame
        DataFrame with NA entries dropped from it.

    See Also
    --------
    DataFrame.isna: Indicate missing values.
    DataFrame.notna : Indicate existing (non-missing) values.
    DataFrame.fillna : Replace missing values.
    Series.dropna : Drop missing values.
    Index.dropna : Drop missing indices.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
    ...                    "toy": [np.nan, 'Batmobile', 'Bullwhip'],
    ...                    "born": [md.NaT, md.Timestamp("1940-04-25"),
    ...                             md.NaT]})
    >>> df.execute()
           name        toy       born
    0    Alfred        NaN        NaT
    1    Batman  Batmobile 1940-04-25
    2  Catwoman   Bullwhip        NaT

    Drop the rows where at least one element is missing.

    >>> df.dropna().execute()
         name        toy       born
    1  Batman  Batmobile 1940-04-25

    Drop the rows where all elements are missing.

    >>> df.dropna(how='all').execute()
           name        toy       born
    0    Alfred        NaN        NaT
    1    Batman  Batmobile 1940-04-25
    2  Catwoman   Bullwhip        NaT

    Keep only the rows with at least 2 non-NA values.

    >>> df.dropna(thresh=2).execute()
           name        toy       born
    1    Batman  Batmobile 1940-04-25
    2  Catwoman   Bullwhip        NaT

    Define in which columns to look for missing values.

    >>> df.dropna(subset=['name', 'born']).execute()
           name        toy       born
    1    Batman  Batmobile 1940-04-25

    Keep the DataFrame with valid entries in the same variable.

    >>> df.dropna(inplace=True)
    >>> df.execute()
         name        toy       born
    1  Batman  Batmobile 1940-04-25
    """
    axis = validate_axis(axis, df)
    use_inf_as_na = pd.get_option("mode.use_inf_as_na")
    if axis != 0:
        raise NotImplementedError("Does not support dropna on DataFrame when axis=1")
    if (
        _drop_na_enable_no_default
        and (how is not no_default)
        and (thresh is not no_default)
    ):
        raise TypeError(
            "You cannot set both the how and thresh arguments at the same time."
        )
    if thresh is no_default and how is no_default:
        how = "any"

    op = DataFrameDropNA(
        axis=axis,
        how=how,
        thresh=thresh,
        subset=subset,
        ignore_index=ignore_index,
        use_inf_as_na=use_inf_as_na,
        output_types=[OutputType.dataframe],
    )
    out_df = op(df)
    if inplace:
        df.data = out_df.data
    else:
        return out_df


def series_dropna(series, axis=0, inplace=False, how=None, ignore_index=False):
    """
    Return a new Series with missing values removed.

    See the :ref:`User Guide <missing_data>` for more on which values are
    considered missing, and how to work with missing data.

    Parameters
    ----------
    axis : {0 or 'index'}, default 0
        There is only one axis to drop values from.
    inplace : bool, default False
        If True, do operation inplace and return None.
    how : str, optional
        Not in use. Kept for compatibility.
    ignore_index : bool, default False
        If True, the resulting axis will be labeled 0, 1, …, n - 1.

    Returns
    -------
    Series
        Series with NA entries dropped from it.

    See Also
    --------
    Series.isna: Indicate missing values.
    Series.notna : Indicate existing (non-missing) values.
    Series.fillna : Replace missing values.
    DataFrame.dropna : Drop rows or columns which contain NA values.
    Index.dropna : Drop missing indices.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> ser = md.Series([1., 2., np.nan])
    >>> ser.execute()
    0    1.0
    1    2.0
    2    NaN
    dtype: float64

    Drop NA values from a Series.

    >>> ser.dropna().execute()
    0    1.0
    1    2.0
    dtype: float64

    Keep the Series with valid entries in the same variable.

    >>> ser.dropna(inplace=True)
    >>> ser.execute()
    0    1.0
    1    2.0
    dtype: float64

    Empty strings are not considered NA values. ``None`` is considered an
    NA value.

    >>> ser = md.Series([np.NaN, '2', md.NaT, '', None, 'I stay'])
    >>> ser.execute()
    0       NaN
    1         2
    2       NaT
    3
    4      None
    5    I stay
    dtype: object
    >>> ser.dropna().execute()
    1         2
    3
    5    I stay
    dtype: object
    """
    axis = validate_axis(axis, series)
    op = DataFrameDropNA(
        axis=axis,
        how=how,
        ignore_index=ignore_index,
        output_types=[OutputType.series],
    )
    out_series = op(series)
    if inplace:
        series.data = out_series.data
    else:
        return out_series


def index_dropna(index, how="any"):
    """
    Return Index without NA/NaN values.

    Parameters
    ----------
    how : {'any', 'all'}, default 'any'
        If the Index is a MultiIndex, drop the value when any or all levels
        are NaN.

    Returns
    -------
    Index
    """
    op = DataFrameDropNA(axis=0, how=how, output_types=[OutputType.index])
    return op(index)
