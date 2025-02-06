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

from typing import Any, List, Optional, Union

import pandas as pd

from ... import opcodes
from ...core import get_output_types
from ...serialization.serializables import BoolField, ListField
from ..core import DataFrame, Index, IndexValue, Series
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class DataFrameReshuffle(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_RESHUFFLE

    group_by = ListField("group_by")
    sort_by = ListField("sort_by")
    ascending = BoolField("ascending", default=None)
    ignore_index = BoolField("ignore_index", default=False)

    def __call__(self, df: Union[DataFrame, Series, Index]):
        if self.ignore_index:
            idx_value = parse_index(pd.RangeIndex(0))
        else:
            idx_value = df.index_value
            if isinstance(idx_value.value, IndexValue.RangeIndex):
                idx_value = parse_index(pd.RangeIndex(1))
        params = df.params
        params["index_value"] = idx_value
        self._output_types = get_output_types(df)
        return self.new_tileable([df], **params)


def df_reshuffle(
    df_obj,
    group_by: Optional[List[Any]] = None,
    sort_by: Optional[List[Any]] = None,
    ascending: bool = True,
    ignore_index: bool = False,
):
    """
    Shuffle data in DataFrame or Series to make data distribution more
    randomized.

    Parameters
    ----------
    group_by: Optional[List[Any]]
        Determine columns to group data while shuffling.
    sort_by: Optional[List[Any]]
    ascending
    ignore_index

    Returns
    -------

    """
    if isinstance(group_by, str):
        group_by = [group_by]
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    if sort_by and not group_by:
        raise ValueError("to use sort_by requires group_by is specified")
    op = DataFrameReshuffle(
        group_by=group_by,
        sort_by=sort_by,
        ascending=ascending,
        ignore_index=ignore_index,
    )
    return op(df_obj)
