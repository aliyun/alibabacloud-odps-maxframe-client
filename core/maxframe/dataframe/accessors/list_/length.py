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

import pandas as pd
import pyarrow as pa

from .... import opcodes
from ....core.entity.output_types import OutputType
from ...operators import DataFrameOperator, DataFrameOperatorMixin


class SeriesListLengthOperator(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.SERIES_LIST_LENGTH

    def __init__(self, **kw):
        super().__init__(_output_types=[OutputType.series], **kw)

    def __call__(self, series):
        return self.new_series(
            [series],
            shape=series.shape,
            index_value=series.index_value,
            dtype=pd.ArrowDtype(pa.int64()),
            name=None,
        )


def series_list_length(series):
    """
    Get the length of each list of the Series.

    Returns
    -------
    Series :
        A Series with data type ``pandas.ArrowDtype(pyarrow.int64)``. Each element
        represents the length of the list, or ``None`` if the list is ``None``.

    Examples
    --------
    Create a series with list type data.

    >>> import maxframe.dataframe as md
    >>> import pyarrow as pa
    >>> from maxframe.lib.dtypes_extension import list_
    >>> s = md.Series(
    ...     data=[[1, 2, 3], [4, 5, 6], None],
    ...     index=[1, 2, 3],
    ...     dtype=list_(pa.int64()),
    ... )
    >>> s.execute()
    1    [1, 2, 3]
    2    [4, 5, 6]
    3         <NA>
    dtype: list<int64>[pyarrow]

    >>> s.list.len().execute()
    1       2
    2       1
    3    <NA>
    dtype: int64[pyarrow]
    """
    return SeriesListLengthOperator()(series)
