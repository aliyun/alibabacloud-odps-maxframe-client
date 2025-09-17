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

from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ...serialization.serializables import AnyField, BoolField, Int32Field, Int64Field
from ..core import DATAFRAME_TYPE, ENTITY_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_df, build_empty_series, parse_index


class BaseDataFrameExpandingAgg(DataFrameOperator, DataFrameOperatorMixin):
    min_periods = Int64Field("min_periods", default=None)
    axis = Int32Field("axis", default=0)
    func = AnyField("func", default=None)

    # always treat count as valid. this behavior is cancelled in pandas 1.0
    count_always_valid = BoolField("count_always_valid", default=None)
    # True if function name is treated as new index
    append_index = BoolField("append_index", default=None)

    def __call__(self, expanding):
        inp = expanding.input
        raw_func = self.func
        self._normalize_funcs()

        if isinstance(inp, DATAFRAME_TYPE):
            empty_df = build_df(inp)
            for c, t in empty_df.dtypes.items():
                if t == np.dtype("O"):
                    empty_df[c] = "O"

            test_df = expanding(empty_df).agg(raw_func)
            if self.axis == 0:
                index_value = inp.index_value
            else:
                index_value = parse_index(
                    test_df.index, expanding.params, inp, store_data=False
                )
            self.append_index = test_df.columns.nlevels != empty_df.columns.nlevels
            return self.new_dataframe(
                [inp],
                shape=(inp.shape[0], test_df.shape[1]),
                dtypes=test_df.dtypes,
                index_value=index_value,
                columns_value=parse_index(test_df.columns, store_data=True),
            )
        else:
            pd_index = inp.index_value.to_pandas()
            empty_series = build_empty_series(
                inp.dtype, index=pd_index[:0], name=inp.name
            )
            test_obj = expanding(empty_series).agg(raw_func)
            if isinstance(test_obj, pd.DataFrame):
                return self.new_dataframe(
                    [inp],
                    shape=(inp.shape[0], test_obj.shape[1]),
                    dtypes=test_obj.dtypes,
                    index_value=inp.index_value,
                    columns_value=parse_index(test_obj.dtypes.index, store_data=True),
                )
            else:
                return self.new_series(
                    [inp],
                    shape=inp.shape,
                    dtype=test_obj.dtype,
                    index_value=inp.index_value,
                    name=test_obj.name,
                )

    def _normalize_funcs(self):
        if isinstance(self.func, dict):
            new_func = OrderedDict()
            for k, v in self.func.items():
                if isinstance(v, str) or callable(v):
                    new_func[k] = [v]
                else:
                    new_func[k] = v
            self.func = new_func
        elif (
            isinstance(self.func, Iterable)
            and not isinstance(self.func, ENTITY_TYPE)
            and not isinstance(self.func, str)
        ):
            self.func = list(self.func)
