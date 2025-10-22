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

from typing import Any

import numpy as np
import pandas as pd

from ... import opcodes
from ... import tensor as mt
from ...core import ENTITY_TYPE, OutputType
from ...serialization.serializables import BoolField
from ...tensor.core import TENSOR_TYPE
from ...utils import get_pd_option
from ..core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE, MultiIndex
from ..operators import DataFrameOperator, DataFrameOperatorMixin


class DataFrameCheckNA(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.CHECK_NA

    positive = BoolField("positive", default=None)
    use_inf_as_na = BoolField("use_inf_as_na", default=None)

    def __call__(self, df):
        if isinstance(df, DATAFRAME_TYPE):
            self.output_types = [OutputType.dataframe]
        elif isinstance(df, SERIES_TYPE):
            self.output_types = [OutputType.series]
        elif isinstance(df, TENSOR_TYPE):
            self.output_types = [OutputType.tensor]
        elif isinstance(df, INDEX_TYPE):
            self.output_types = [OutputType.index]
        else:
            raise TypeError(
                f"Expecting maxframe dataframe, series, index, or tensor, got {type(df)}"
            )

        params = df.params
        if self.output_types[0] == OutputType.dataframe:
            if df.dtypes is None:
                params["dtypes"] = None
            else:
                params["dtypes"] = pd.Series(
                    [np.dtype("bool")] * len(df.dtypes),
                    index=df.columns_value.to_pandas(),
                )
        else:
            params["dtype"] = np.dtype("bool")
        return self.new_tileable([df], **params)


def _from_pandas(obj: Any):
    if isinstance(obj, pd.DataFrame):
        from ..datasource.dataframe import from_pandas

        return from_pandas(obj)
    elif isinstance(obj, pd.Series):
        from ..datasource.series import from_pandas

        return from_pandas(obj)
    elif isinstance(obj, np.ndarray):
        return mt.tensor(obj)
    else:
        return obj


def isna(obj):
    """
    Detect missing values.

    Return a boolean same-sized object indicating if the values are NA.
    NA values, such as None or :attr:`numpy.NaN`, gets mapped to True
    values.

    Everything else gets mapped to False values. Characters such as empty
    strings ``''`` or :attr:`numpy.inf` are not considered NA values
    (unless you set ``pandas.options.mode.use_inf_as_na = True``).

    Returns
    -------
    DataFrame
        Mask of bool values for each element in DataFrame that
        indicates whether an element is not an NA value.

    See Also
    --------
    DataFrame.isnull : Alias of isna.
    DataFrame.notna : Boolean inverse of isna.
    DataFrame.dropna : Omit axes labels with missing values.
    isna : Top-level isna.

    Examples
    --------
    Show which entries in a DataFrame are NA.

    >>> import numpy as np
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'age': [5, 6, np.NaN],
    ...                    'born': [md.NaT, md.Timestamp('1939-05-27'),
    ...                             md.Timestamp('1940-04-25')],
    ...                    'name': ['Alfred', 'Batman', ''],
    ...                    'toy': [None, 'Batmobile', 'Joker']})
    >>> df.execute()
       age       born    name        toy
    0  5.0        NaT  Alfred       None
    1  6.0 1939-05-27  Batman  Batmobile
    2  NaN 1940-04-25              Joker

    >>> df.isna().execute()
         age   born   name    toy
    0  False   True  False   True
    1  False  False  False  False
    2   True  False  False  False

    Show which entries in a Series are NA.

    >>> ser = md.Series([5, 6, np.NaN])
    >>> ser.execute()
    0    5.0
    1    6.0
    2    NaN
    dtype: float64

    >>> ser.isna().execute()
    0    False
    1    False
    2     True
    dtype: bool
    """
    use_inf_as_na = get_pd_option("mode.use_inf_as_na", False)
    if isinstance(obj, MultiIndex):
        raise NotImplementedError("isna is not defined for MultiIndex")
    elif isinstance(obj, ENTITY_TYPE):
        if isinstance(obj, TENSOR_TYPE):
            return mt.isnan(obj)
        else:
            op = DataFrameCheckNA(positive=True, use_inf_as_na=use_inf_as_na)
            return op(obj)
    else:
        return _from_pandas(pd.isna(obj))


def notna(obj):
    """
    Detect existing (non-missing) values.

    Return a boolean same-sized object indicating if the values are not NA.
    Non-missing values get mapped to True. Characters such as empty
    strings ``''`` or :attr:`numpy.inf` are not considered NA values
    (unless you set ``pandas.options.mode.use_inf_as_na = True``).
    NA values, such as None or :attr:`numpy.NaN`, get mapped to False
    values.

    Returns
    -------
    DataFrame
        Mask of bool values for each element in DataFrame that
        indicates whether an element is not an NA value.

    See Also
    --------
    DataFrame.notnull : Alias of notna.
    DataFrame.isna : Boolean inverse of notna.
    DataFrame.dropna : Omit axes labels with missing values.
    notna : Top-level notna.

    Examples
    --------
    Show which entries in a DataFrame are not NA.

    >>> import numpy as np
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'age': [5, 6, np.NaN],
    ...                    'born': [md.NaT, md.Timestamp('1939-05-27'),
    ...                             md.Timestamp('1940-04-25')],
    ...                    'name': ['Alfred', 'Batman', ''],
    ...                    'toy': [None, 'Batmobile', 'Joker']})
    >>> df.execute()
       age       born    name        toy
    0  5.0        NaT  Alfred       None
    1  6.0 1939-05-27  Batman  Batmobile
    2  NaN 1940-04-25              Joker

    >>> df.notna().execute()
         age   born  name    toy
    0   True  False  True  False
    1   True   True  True   True
    2  False   True  True   True

    Show which entries in a Series are not NA.

    >>> ser = md.Series([5, 6, np.NaN])
    >>> ser.execute()
    0    5.0
    1    6.0
    2    NaN
    dtype: float64

    >>> ser.notna().execute()
    0     True
    1     True
    2    False
    dtype: bool
    """
    use_inf_as_na = get_pd_option("mode.use_inf_as_na", False)
    if isinstance(obj, MultiIndex):
        raise NotImplementedError("isna is not defined for MultiIndex")
    elif isinstance(obj, ENTITY_TYPE):
        if isinstance(obj, TENSOR_TYPE):
            return ~mt.isnan(obj)
        else:
            op = DataFrameCheckNA(positive=False, use_inf_as_na=use_inf_as_na)
            return op(obj)
    else:
        return _from_pandas(pd.notna(obj))


isnull = isna
notnull = notna
