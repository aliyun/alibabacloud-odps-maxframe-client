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
from ...serialization.serializables import AnyField, StringField
from ...utils import no_default
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import make_column_list


class DataFrameCollectKv(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.COLLECT_KV

    columns = AnyField("columns", default=None)
    kv_delim = StringField("kv_delim", default=None)
    item_delim = StringField("item_delim", default=None)
    kv_col = StringField("kv_col", default=None)

    def __call__(self, df):
        if self.columns is None:
            cols = list(df.dtypes.index)
        else:
            cols = self.columns if isinstance(self.columns, list) else [self.columns]
        new_dtypes = df.dtypes.drop(cols, errors="ignore")
        new_dtypes = pd.concat(
            [new_dtypes, pd.Series([np.dtype("object")], index=[self.kv_col])]
        )
        shape = (df.shape[0], len(new_dtypes))
        return self.new_dataframe(
            [df],
            shape=shape,
            dtypes=new_dtypes,
            index_value=df.index_value,
            columns_value=new_dtypes.index,
        )


def collect_kv(
    data,
    columns=None,
    kv_delim="=",
    item_delim=",",
    kv_col="kv_col",
):
    """
    Merge values in specified columns into a key-value represented column.

    Parameters
    ----------
    columns : list, default None
        The columns to be merged.
    kv_delim : str, default '='
        Delimiter between key and value.
    item_delim : str, default ','
        Delimiter between key-value pairs.
    kv_col : str, default 'kv_col'
        Name of the new key-value column

    Returns
    -------
    DataFrame
        converted data frame

    See Also
    --------
    DataFrame.mf.extract_kv

    Examples
    -------
    >>> import maxframe.dataframe as md

    >>> df = md.DataFrame({"name": ["name1", "name2", "name3", "name4", "name5"],
    ...                    "k1": [1.0, NaN, 7.1, NaN, NaN],
    ...                    "k2": [3.0, 3.0, NaN, 1.2, 1.0],
    ...                    "k3": [NaN, 5.1, NaN, 1.5, NaN],
    ...                    "k5": [10.0, NaN, NaN, NaN, NaN,],
    ...                    "k7": [NaN, NaN, 8.2, NaN, NaN, ],
    ...                    "k9": [NaN, NaN, NaN, NaN, 1.1]})
    >>> df.execute()
       name   k1   k2   k3   k5    k7   k9
    0  name1  1.0  3.0  NaN  10.0  NaN  NaN
    1  name2  NaN  3.0  5.1  NaN   NaN  NaN
    2  name3  7.1  NaN  NaN  NaN   8.2  NaN
    3  name4  NaN  1.2  1.5  NaN   NaN  NaN
    4  name5  NaN  1.0  NaN  NaN   NaN  1.1

    The field names to be merged are specified by columns
    kv_delim is to delimit the key and value and '=' is default
    item_delim is to delimit the Key-Value pairs, ',' is default
    The new column name is specified by kv_col, 'kv_col' is default

    >>> df.mf.collect_kv(columns=['k1', 'k2', 'k3', 'k5', 'k7', 'k9']).execute()
       name   kv_col
    0  name1  k1=1.0,k2=3.0,k5=10.0
    1  name2  k2=3.0,k3=5.1
    2  name3  k1=7.1,k7=8.2
    3  name4  k2=1.2,k3=1.5
    4  name5  k2=1.0,k9=1.1
    """
    columns_list = make_column_list(columns, data.dtypes) or []
    non_exist_key = next(
        (c for c in columns_list if c not in data.dtypes.index), no_default
    )
    if columns_list and non_exist_key is not no_default:
        raise ValueError(f"Column {non_exist_key} specified is not a valid column.")
    op = DataFrameCollectKv(
        columns=columns,
        kv_delim=kv_delim,
        item_delim=item_delim,
        kv_col=kv_col,
    )
    return op(data)
