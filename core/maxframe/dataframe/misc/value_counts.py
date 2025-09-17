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
from ...serialization.serializables import BoolField, Int64Field, KeyField, StringField
from ...utils import pd_release_version
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_df, build_series, parse_index

_keep_original_order = pd_release_version >= (1, 3, 0)
_name_count_or_proportion = pd_release_version >= (2, 0, 0)


class DataFrameValueCounts(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.VALUE_COUNTS

    input = KeyField("input")
    normalize = BoolField("normalize", default=False)
    sort = BoolField("sort", default=True)
    ascending = BoolField("ascending", default=False)
    bins = Int64Field("bins", default=None)
    dropna = BoolField("dropna", default=True)
    method = StringField("method", default=None)
    convert_index_to_interval = BoolField("convert_index_to_interval", default=None)
    nrows = Int64Field("nrows", default=None)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.output_types = [OutputType.series]

    @classmethod
    def _set_inputs(cls, op: "DataFrameValueCounts", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.input = op._inputs[0]

    def __call__(self, inp):
        if inp.ndim == 2:
            idx = pd.MultiIndex.from_frame(build_df(inp))
            test_series = build_series(index=idx, dtype=int)
            if _name_count_or_proportion:
                out_name = "proportion" if self.normalize else "count"
            else:
                out_name = None
        else:
            test_series = build_series(inp).value_counts(normalize=self.normalize)
            out_name = test_series.name
        if self.bins is not None:
            from .cut import cut

            # cut
            try:
                inp = cut(inp, self.bins, include_lowest=True)
            except TypeError:  # pragma: no cover
                raise TypeError("bins argument only works with numeric data.")

            self.bins = None
            self.convert_index_to_interval = True
            return self.new_series(
                [inp],
                shape=(np.nan,),
                index_value=parse_index(pd.CategoricalIndex([]), inp, store_data=False),
                name=out_name,
                dtype=test_series.dtype,
            )
        else:
            return self.new_series(
                [inp],
                shape=(np.nan,),
                index_value=parse_index(test_series.index, store_data=False),
                name=out_name,
                dtype=test_series.dtype,
            )


def value_counts(
    series,
    normalize=False,
    sort=True,
    ascending=False,
    bins=None,
    dropna=True,
    method="auto",
):
    # FIXME: https://github.com/aliyun/alibabacloud-odps-maxframe-client/issues/33
    """
    Return a Series containing counts of unique values.

    The resulting object will be in descending order so that the
    first element is the most frequently-occurring element.
    Excludes NA values by default.

    Parameters
    ----------
    normalize : bool, default False
        If True then the object returned will contain the relative
        frequencies of the unique values.
    sort : bool, default True
        Sort by frequencies.
    ascending : bool, default False
        Sort in ascending order.
    bins : int, optional
        Rather than count values, group them into half-open bins,
        a convenience for ``pd.cut``, only works with numeric data.
    dropna : bool, default True
        Don't include counts of NaN.
    method : str, default 'auto'
        'auto', 'shuffle', or 'tree', 'tree' method provide
        a better performance, while 'shuffle' is recommended
        if aggregated result is very large, 'auto' will use
        'shuffle' method in distributed mode and use 'tree'
        in local mode.

    Returns
    -------
    Series

    See Also
    --------
    Series.count: Number of non-NA elements in a Series.
    DataFrame.count: Number of non-NA elements in a DataFrame.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> import numpy as np
    >>> s = md.Series([3, 1, 2, 3, 4, np.nan])
    >>> s.value_counts().execute()
    3.0    2
    4.0    1
    2.0    1
    1.0    1
    dtype: int64

    With `normalize` set to `True`, returns the relative frequency by
    dividing all values by the sum of values.

    >>> s = md.Series([3, 1, 2, 3, 4, np.nan])
    >>> s.value_counts(normalize=True).execute()
    3.0    0.4
    4.0    0.2
    2.0    0.2
    1.0    0.2
    dtype: float64

    **dropna**

    With `dropna` set to `False` we can also see NaN index values.

    >>> s.value_counts(dropna=False).execute()
    3.0    2
    NaN    1
    4.0    1
    2.0    1
    1.0    1
    dtype: int64
    """
    op = DataFrameValueCounts(
        normalize=normalize,
        sort=sort,
        ascending=ascending,
        bins=bins,
        dropna=dropna,
        method=method,
    )
    return op(series)


def df_value_counts(
    df,
    subset=None,
    normalize=False,
    sort=True,
    ascending=False,
    dropna=True,
    method="auto",
):
    if not subset:
        df_to_count = df
    else:
        df_to_count = df[subset]

    op = DataFrameValueCounts(
        normalize=normalize,
        sort=sort,
        ascending=ascending,
        dropna=dropna,
        method=method,
    )
    return op(df_to_count)
