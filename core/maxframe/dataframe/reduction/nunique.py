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

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

from ... import opcodes
from ...config import options
from ...core import OutputType
from ...serialization.serializables import BoolField
from ...utils import lazy_import
from .core import DataFrameReductionMixin, DataFrameReductionOperator

cudf = lazy_import("cudf")


class DataFrameNunique(DataFrameReductionOperator, DataFrameReductionMixin):
    _op_type_ = opcodes.NUNIQUE
    _func_name = "nunique"

    dropna = BoolField("dropna", default=None)
    use_arrow_dtype = BoolField("use_arrow_dtype", default=None)

    @property
    def is_atomic(self):
        return True

    @classmethod
    def get_reduction_callable(cls, op):
        func_name = getattr(op, "_func_name")
        kw = dict(dropna=op.dropna)
        kw = {k: v for k, v in kw.items() if v is not None}

        def fun(value):
            return value.nunique(**kw)

        fun.__name__ = func_name
        return fun


def nunique_dataframe(df, axis=0, dropna=True):
    """
    Count distinct observations over requested axis.

    Return Series with number of distinct observations. Can ignore NaN
    values.

    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for
        column-wise.
    dropna : bool, default True
        Don't include NaN in the counts.

    Returns
    -------
    Series

    See Also
    --------
    Series.nunique: Method nunique for Series.
    DataFrame.count: Count non-NA cells for each column or row.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> df.nunique().execute()
    A    3
    B    1
    dtype: int64

    >>> df.nunique(axis=1).execute()
    0    1
    1    2
    2    2
    dtype: int64
    """
    op = DataFrameNunique(
        axis=axis,
        dropna=dropna,
        output_types=[OutputType.series],
    )
    return op(df)


def nunique_series(series, dropna=True):
    """
    Return number of unique elements in the object.

    Excludes NA values by default.

    Parameters
    ----------
    dropna : bool, default True
        Don't include NaN in the count.

    Returns
    -------
    int

    See Also
    --------
    DataFrame.nunique: Method nunique for DataFrame.
    Series.count: Count non-NA/null observations in the Series.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s = md.Series([1, 3, 5, 7, 7])
    >>> s.execute()
    0    1
    1    3
    2    5
    3    7
    4    7
    dtype: int64

    >>> s.nunique().execute()
    4
    """
    op = DataFrameNunique(
        dropna=dropna,
        output_types=[OutputType.scalar],
        use_arrow_dtype=options.dataframe.use_arrow_dtype,
    )
    return op(series)
