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

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import AnyField, BoolField
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_empty_df, parse_index


class DataFrameSetIndex(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_SET_INDEX

    keys = AnyField("keys")
    drop = BoolField("drop", default=True)
    append = BoolField("append", default=False)
    verify_integrity = BoolField("verify_integrity", default=False)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def __call__(self, df):
        new_df = build_empty_df(df.dtypes, index=df.index_value.to_pandas()).set_index(
            keys=self.keys,
            drop=self.drop,
            append=self.append,
            verify_integrity=self.verify_integrity,
        )
        return self.new_dataframe(
            [df],
            shape=(df.shape[0], new_df.shape[1]),
            dtypes=new_df.dtypes,
            index_value=parse_index(new_df.index),
            columns_value=parse_index(new_df.columns, store_data=True),
        )


def set_index(df, keys, drop=True, append=False, inplace=False, verify_integrity=False):
    # TODO add support for set index by series, index, mt.ndarray, etc.
    """
    Set the DataFrame index using existing columns.

    Set the DataFrame index (row labels) using one or more existing
    columns. The index can replace the existing index or expand on it.

    Parameters
    ----------
    keys : label or array-like or list of labels
        This parameter can be either a single column key, or a list containing column keys.
    drop : bool, default True
        Delete columns to be used as the new index.
    append : bool, default False
        Whether to append columns to existing index.
    inplace : bool, default False
        If True, modifies the DataFrame in place (do not create a new object).
    verify_integrity : bool, default False
        Check the new index for duplicates. Otherwise defer the check until
        necessary. Setting to False will improve the performance of this
        method.

    Returns
    -------
    DataFrame or None
        Changed row labels or None if ``inplace=True``.

    See Also
    --------
    DataFrame.reset_index : Opposite of set_index.
    DataFrame.reindex : Change to new indices or expand indices.
    DataFrame.reindex_like : Change to same indices as other DataFrame.

    Examples
    --------
    >>> import maxframe.dataframe as md

    >>> df = md.DataFrame({'month': [1, 4, 7, 10],
    ...                    'year': [2012, 2014, 2013, 2014],
    ...                    'sale': [55, 40, 84, 31]})
    >>> df
       month  year  sale
    0      1  2012    55
    1      4  2014    40
    2      7  2013    84
    3     10  2014    31

    Set the index to become the 'month' column:

    >>> df.set_index('month')
           year  sale
    month
    1      2012    55
    4      2014    40
    7      2013    84
    10     2014    31

    Create a MultiIndex using columns 'year' and 'month':

    >>> df.set_index(['year', 'month'])
                sale
    year  month
    2012  1     55
    2014  4     40
    2013  7     84
    2014  10    31
    """
    op = DataFrameSetIndex(
        keys=keys,
        drop=drop,
        append=append,
        verify_integrity=verify_integrity,
        output_types=[OutputType.dataframe],
    )
    result = op(df)
    if not inplace:
        return result
    else:
        df.data = result.data
