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
import pandas as pd

from ... import opcodes
from ...core import EntityData, OutputType
from ...serialization.serializables import AnyField, KeyField, StringField
from ...utils import make_dtype, no_default
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import make_column_list


class DataFrameExtractKv(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.EXTRACT_KV

    columns = AnyField("columns", default=None)
    kv_delim = StringField("kv_delim", default="=")
    item_delim = StringField("item_delim", default=",")
    dtype = AnyField("dtype", default=None)
    fill_value = AnyField("fill_value", default=None)
    errors = StringField("errors", default="raise")
    # intermediate agg data
    agg_results = KeyField("agg_results", default=None)

    def __init__(self, kv_delim="=", item_delim=",", **kw):
        super().__init__(kv_delim=kv_delim, item_delim=item_delim, **kw)
        self.output_types = [OutputType.dataframe]

    @classmethod
    def _set_inputs(cls, op: "DataFrameExtractKv", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if op.agg_results is not None:
            op.agg_results = inputs[-1]

    def __call__(self, df):
        shape = (df.shape[0], np.nan)
        errors_arg = self.errors

        def get_keys(row, cols, kv_delim, item_delim):
            for col in cols:
                if row[col] is not None:
                    pairs = row[col].split(item_delim)
                else:
                    pairs = []
                for pair in pairs:
                    result = pair.split(kv_delim, 1)
                    if len(result) == 2:
                        yield f"{col}_{result[0]}"
                    elif errors_arg == "raise":
                        raise ValueError(f"Malformed data {pair} in column '{col}'.")

        all_keys = df.mf.flatmap(
            get_keys,
            dtypes=pd.Series([str], index=["keys_cols"]),
            cols=self.columns,
            kv_delim=self.kv_delim,
            item_delim=self.item_delim,
        )
        self.agg_results = all_keys.drop_duplicates().sort_values(by="keys_cols")
        inputs = [df]
        inputs.append(self.agg_results)
        return self.new_dataframe(
            inputs,
            shape=shape,
            dtypes=None,
            index_value=df.index_value,
            columns_value=None,
        )


def extract_kv(
    data,
    columns=None,
    kv_delim="=",
    item_delim=",",
    dtype="float",
    fill_value=None,
    errors="raise",
):
    """
    Extract values in key-value represented columns into standalone columns.
    New column names will be the name of the key-value column followed by
    an underscore and the key.

    Parameters
    ----------
    columns : list, default None
        The key-value columns to be extracted.
    kv_delim : str, default '='
        Delimiter between key and value.
    item_delim : str, default ','
        Delimiter between key-value pairs.
    dtype : str
        Type of value columns to generate.
    fill_value : object, default None
        Default value for missing key-value pairs.
    errors : {'ignore', 'raise'}, default 'raise'
        * If 'raise', then invalid parsing will raise an exception.
        * If 'ignore', then invalid parsing will return the input.

    Returns
    -------
    DataFrame
        extracted data frame

    See Also
    --------
    DataFrame.mf.collect_kv

    Examples
    --------
    >>> import numpy as np
    >>> import maxframe.dataframe as md

    >>> df = md.DataFrame({"name": ["name1", "name2", "name3", "name4", "name5"],
    ...                    "kv": ["k1=1.0,k2=3.0,k5=10.0",
    ...                           "k2=3.0,k3=5.1",
    ...                           "k1=7.1,k7=8.2",
    ...                           "k2=1.2,k3=1.5",
    ...                           "k2=1.0,k9=1.1"]})
    >>> df.execute()
       name   kv
    0  name1  k1=1.0,k2=3.0,k5=10.0
    1  name2  k2=3.0,k3=5.1
    2  name3  k1=7.1,k7=8.2
    3  name4  k2=1.2,k3=1.5
    4  name5  k2=1.0,k9=1.1

    The field names to be expanded are specified by columns
    kv_delim is to delimit the key and value and '=' is default
    item_delim is to delimit the Key-Value pairs, ',' is default
    The output field name is the original field name connect with the key by "_"
    fill_value is used to fill missing values, None is default

    >>> df.mf.extract_kv(columns=['kv'], kv_delim='=', item_delim=',').execute()
       name   kv_k1   kv_k2   kv_k3   kv_k5   kv_k7   kv_k9
    0  name1  1.0     3.0     NaN     10.0    NaN     NaN
    1  name2  NaN     3.0     5.1     NaN     NaN     NaN
    2  name3  7.1     NaN     NaN     NaN     8.2     NaN
    3  name4  NaN     1.2     1.5     NaN     NaN     NaN
    4  name5  NaN     1.0     NaN     NaN     NaN     1.1
    """
    if columns is None:
        columns = data.dtypes.index.tolist()
    columns_list = make_column_list(columns, data.dtypes)
    non_exist_key = next(
        (c for c in columns_list if c not in data.dtypes.index), no_default
    )
    if non_exist_key is not no_default:
        raise ValueError(f"Column {non_exist_key} specified is not a valid column.")
    for col in columns_list:
        if str(data.dtypes[col]) not in ("object", "string"):
            raise ValueError(f"Column '{col}' must be of string type.")
    op = DataFrameExtractKv(
        columns=columns,
        kv_delim=kv_delim,
        item_delim=item_delim,
        dtype=make_dtype(dtype),
        fill_value=fill_value,
        errors=errors,
    )
    return op(data)
