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

import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, Entity, EntityData, get_output_types
from ...serialization.serializables import AnyField, Int64Field, StringField
from ..core import DATAFRAME_TYPE, SERIES_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import validate_axis


class DataFrameFillNA(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.FILL_NA
    _legacy_name = "FillNA"  # since v2.0.0

    value = AnyField(
        "value", on_serialize=lambda x: x.data if isinstance(x, Entity) else x
    )
    method = StringField("method", default=None)
    axis = AnyField("axis", default=0)
    limit = Int64Field("limit", default=None)
    downcast = AnyField("downcast", default=None)

    def __init__(self, output_limit=1, output_types=None, **kw):
        super().__init__(output_limit=output_limit, _output_types=output_types, **kw)

    @classmethod
    def _set_inputs(cls, op: "DataFrameFillNA", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if op.method is None and len(inputs) > 1:
            op.value = op._inputs[1]

    def __call__(self, a, value_df=None):
        method = getattr(self, "method", None)
        if method == "backfill":
            method = "bfill"
        elif method == "pad":
            method = "ffill"
        self.method = method
        axis = getattr(self, "axis", None) or 0
        self.axis = validate_axis(axis, a)

        inputs = [a]
        if value_df is not None:
            inputs.append(value_df)
        if isinstance(a, DATAFRAME_TYPE):
            return self.new_dataframe(
                inputs,
                shape=a.shape,
                dtypes=a.dtypes,
                index_value=a.index_value,
                columns_value=a.columns_value,
            )
        elif isinstance(a, SERIES_TYPE):
            return self.new_series(
                inputs,
                shape=a.shape,
                dtype=a.dtype,
                index_value=a.index_value,
                name=a.name,
            )
        else:
            return self.new_index(
                inputs,
                shape=a.shape,
                dtype=a.dtype,
                index_value=a.index_value,
                name=a.name,
                names=a.names,
            )


# keep for import compatibility
FillNA = DataFrameFillNA


def fillna(
    df, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None
):
    """
    Fill NA/NaN values using the specified method.

    Parameters
    ----------
    value : scalar, dict, Series, or DataFrame
        Value to use to fill holes (e.g. 0), alternately a
        dict/Series/DataFrame of values specifying which value to use for
        each index (for a Series) or column (for a DataFrame).  Values not
        in the dict/Series/DataFrame will not be filled. This value cannot
        be a list.
    method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
        Method to use for filling holes in reindexed Series
        pad / ffill: propagate last valid observation forward to next valid
        backfill / bfill: use next valid observation to fill gap.
    axis : {0 or 'index', 1 or 'columns'}
        Axis along which to fill missing values.
    inplace : bool, default False
        If True, fill in-place. Note: this will modify any
        other views on this object (e.g., a no-copy slice for a column in a
        DataFrame).
    limit : int, default None
        If method is specified, this is the maximum number of consecutive
        NaN values to forward/backward fill. In other words, if there is
        a gap with more than this number of consecutive NaNs, it will only
        be partially filled. If method is not specified, this is the
        maximum number of entries along the entire axis where NaNs will be
        filled. Must be greater than 0 if not None.
    downcast : dict, default is None
        A dict of item->dtype of what to downcast if possible,
        or the string 'infer' which will try to downcast to an appropriate
        equal type (e.g. float64 to int64 if possible).

    Returns
    -------
    DataFrame or None
        Object with missing values filled or None if ``inplace=True``.

    See Also
    --------
    interpolate : Fill NaN values using interpolation.
    reindex : Conform object to new index.
    asfreq : Convert TimeSeries to specified frequency.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([[np.nan, 2, np.nan, 0],
                           [3, 4, np.nan, 1],
                           [np.nan, np.nan, np.nan, 5],
                           [np.nan, 3, np.nan, 4]],
                          columns=list('ABCD'))
    >>> df.execute()
         A    B   C  D
    0  NaN  2.0 NaN  0
    1  3.0  4.0 NaN  1
    2  NaN  NaN NaN  5
    3  NaN  3.0 NaN  4

    Replace all NaN elements with 0s.

    >>> df.fillna(0).execute()
        A   B   C   D
    0   0.0 2.0 0.0 0
    1   3.0 4.0 0.0 1
    2   0.0 0.0 0.0 5
    3   0.0 3.0 0.0 4

    We can also propagate non-null values forward or backward.

    >>> df.fillna(method='ffill').execute()
        A   B   C   D
    0   NaN 2.0 NaN 0
    1   3.0 4.0 NaN 1
    2   3.0 4.0 NaN 5
    3   3.0 3.0 NaN 4

    Replace all NaN elements in column 'A', 'B', 'C', and 'D', with 0, 1,
    2, and 3 respectively.

    >>> values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    >>> df.fillna(value=values).execute()
        A   B   C   D
    0   0.0 2.0 2.0 0
    1   3.0 4.0 2.0 1
    2   0.0 1.0 2.0 5
    3   0.0 3.0 2.0 4
    """
    if value is None and method is None:
        raise ValueError("Must specify a fill 'value' or 'method'.")
    elif value is not None and method is not None:
        raise ValueError("Cannot specify both 'value' and 'method'.")

    if isinstance(df, SERIES_TYPE) and isinstance(
        value, (DATAFRAME_TYPE, pd.DataFrame)
    ):
        raise ValueError(
            '"value" parameter must be a scalar, dict or Series, but you passed a "%s"'
            % type(value).__name__
        )

    if downcast is not None:
        raise NotImplementedError(
            'Currently argument "downcast" is not implemented yet'
        )
    if limit is not None:
        raise NotImplementedError('Currently argument "limit" is not implemented yet')

    if isinstance(value, ENTITY_TYPE):
        value, value_df = None, value
    else:
        value_df = None

    op = DataFrameFillNA(
        value=value,
        method=method,
        axis=axis,
        limit=limit,
        downcast=downcast,
        output_types=get_output_types(df),
    )
    out_df = op(df, value_df=value_df)
    if inplace:
        df.data = out_df.data
    else:
        return out_df


def ffill(df, axis=None, inplace=False, limit=None, downcast=None):
    """
    Synonym for :meth:`DataFrame.fillna` with ``method='ffill'``.

    Returns
    -------
    {klass} or None
        Object with missing values filled or None if ``inplace=True``.
    """
    return fillna(
        df, method="ffill", axis=axis, inplace=inplace, limit=limit, downcast=downcast
    )


def bfill(df, axis=None, inplace=False, limit=None, downcast=None):
    """
    Synonym for :meth:`DataFrame.fillna` with ``method='bfill'``.

    Returns
    -------
    {klass} or None
        Object with missing values filled or None if ``inplace=True``.
    """
    return fillna(
        df, method="bfill", axis=axis, inplace=inplace, limit=limit, downcast=downcast
    )


def index_fillna(index, value=None, downcast=None):
    """
    Fill NA/NaN values with the specified value.

    Parameters
    ----------
    value : scalar
        Scalar value to use to fill holes (e.g. 0).
        This value cannot be a list-likes.
    downcast : dict, default is None
        A dict of item->dtype of what to downcast if possible,
        or the string 'infer' which will try to downcast to an appropriate
        equal type (e.g. float64 to int64 if possible).

    Returns
    -------
    Index

    See Also
    --------
    DataFrame.fillna : Fill NaN values of a DataFrame.
    Series.fillna : Fill NaN Values of a Series.
    """
    if isinstance(value, (list, pd.Series, SERIES_TYPE)):
        raise ValueError("'value' must be a scalar, passed: %s" % type(value))

    op = DataFrameFillNA(
        value=value,
        downcast=downcast,
        output_types=get_output_types(index),
    )
    return op(index)
