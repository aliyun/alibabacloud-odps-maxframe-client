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
from ...serialization.serializables import AnyField
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class IndexGetLevelValues(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.GET_LEVEL_VALUES

    level = AnyField("level")

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def __call__(self, index):
        empty_index = index.index_value.to_pandas()
        result_index = empty_index.get_level_values(self.level)

        return self.new_index(
            [index],
            shape=(index.shape[0],),
            dtype=result_index.dtype,
            index_value=parse_index(result_index, store_data=False),
            names=result_index.names,
        )


def get_level_values(index, level):
    """
    Return vector of label values for requested level.

    Length of returned vector is equal to the length of the index.

    Parameters
    ----------
    level : int or str
        ``level`` is either the integer position of the level in the
        MultiIndex, or the name of the level.

    Returns
    -------
    values : Index
        Values is a level of this MultiIndex converted to
        a single :class:`Index` (or subclass thereof).

    Examples
    --------
    Create a MultiIndex:

    >>> import maxframe.dataframe as md
    >>> import pandas as pd
    >>> mi = md.Index(pd.MultiIndex.from_arrays((list('abc'), list('def')), names=['level_1', 'level_2']))

    Get level values by supplying level as either integer or name:

    >>> mi.get_level_values(0).execute()
    Index(['a', 'b', 'c'], dtype='object', name='level_1')
    >>> mi.get_level_values('level_2').execute()
    Index(['d', 'e', 'f'], dtype='object', name='level_2')
    """
    op = IndexGetLevelValues(level=level)
    return op(index)
