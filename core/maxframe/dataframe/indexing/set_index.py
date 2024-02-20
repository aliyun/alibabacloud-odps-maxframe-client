# Copyright 1999-2024 Alibaba Group Holding Ltd.
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
        new_df = build_empty_df(df.dtypes).set_index(
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
