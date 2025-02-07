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

from ... import opcodes
from ...serialization.serializables import BoolField, Int64Field
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class DataFrameMemoryUsage(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.MEMORY_USAGE

    # raw arguments of memory_usage method
    index = BoolField("index", default=None)
    deep = BoolField("deep", default=None)

    # size of range index, when set, the value will be prepended to the result series
    # if the input is a dataframe, or added to the result when the input is a series
    range_index_size = Int64Field("range_index_size", default=None)

    def _adapt_index(self, input_index, index=0):
        """
        When ``index=True`` is passed, an extra column will be prepended to the result series
        Thus we need to update the index of initial chunk for returned dataframe chunks
        """
        if not self.index or index != 0:
            return input_index
        idx_data = input_index.to_pandas().insert(0, "Index")
        return parse_index(idx_data, store_data=True)

    def _adapt_nsplits(self, input_nsplit):
        """
        When ``index=True`` is passed, the size of returned series is one element larger
        than the number of columns, which affects ``nsplits``.
        """
        if not self.index:
            return (input_nsplit[-1],)
        nsplits_list = list(input_nsplit[-1])
        nsplits_list[0] += 1
        return (tuple(nsplits_list),)

    def __call__(self, df_or_series):
        """
        Return output object of memory_usage() call
        """
        if df_or_series.ndim == 1:
            # the input data is a series, a Scalar will be returned
            return self.new_scalar([df_or_series], dtype=np.dtype(int))
        else:
            # the input data is a DataFrame, a Scalar will be returned
            # calculate shape of returning series given ``op.index``
            new_shape = (
                (df_or_series.shape[-1] + 1,)
                if self.index
                else (df_or_series.shape[-1],)
            )
            return self.new_series(
                [df_or_series],
                index_value=self._adapt_index(df_or_series.columns_value),
                shape=new_shape,
                dtype=np.dtype(int),
            )


def df_memory_usage(df, index=True, deep=False):
    """
    Return the memory usage of each column in bytes.

    The memory usage can optionally include the contribution of
    the index and elements of `object` dtype.

    This value is displayed in `DataFrame.info` by default. This can be
    suppressed by setting ``pandas.options.display.memory_usage`` to False.

    Parameters
    ----------
    index : bool, default True
        Specifies whether to include the memory usage of the DataFrame's
        index in returned Series. If ``index=True``, the memory usage of
        the index is the first item in the output.
    deep : bool, default False
        If True, introspect the data deeply by interrogating
        `object` dtypes for system-level memory consumption, and include
        it in the returned values.

    Returns
    -------
    Series
        A Series whose index is the original column names and whose values
        is the memory usage of each column in bytes.

    See Also
    --------
    numpy.ndarray.nbytes : Total bytes consumed by the elements of an
        ndarray.
    Series.memory_usage : Bytes consumed by a Series.
    Categorical : Memory-efficient array for string values with
        many repeated values.
    DataFrame.info : Concise summary of a DataFrame.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> dtypes = ['int64', 'float64', 'complex128', 'object', 'bool']
    >>> data = dict([(t, mt.ones(shape=5000).astype(t))
    ...              for t in dtypes])
    >>> df = md.DataFrame(data)
    >>> df.head().execute()
       int64  float64            complex128  object  bool
    0      1      1.0    1.000000+0.000000j       1  True
    1      1      1.0    1.000000+0.000000j       1  True
    2      1      1.0    1.000000+0.000000j       1  True
    3      1      1.0    1.000000+0.000000j       1  True
    4      1      1.0    1.000000+0.000000j       1  True

    >>> df.memory_usage().execute()
    Index           128
    int64         40000
    float64       40000
    complex128    80000
    object        40000
    bool           5000
    dtype: int64

    >>> df.memory_usage(index=False).execute()
    int64         40000
    float64       40000
    complex128    80000
    object        40000
    bool           5000
    dtype: int64

    The memory footprint of `object` dtype columns is ignored by default:

    >>> df.memory_usage(deep=True).execute()
    Index            128
    int64          40000
    float64        40000
    complex128     80000
    object        160000
    bool            5000
    dtype: int64

    Use a Categorical for efficient storage of an object-dtype column with
    many repeated values.

    >>> df['object'].astype('category').memory_usage(deep=True).execute()
    5216
    """
    op = DataFrameMemoryUsage(index=index, deep=deep)
    return op(df)


def series_memory_usage(series, index=True, deep=False):
    """
    Return the memory usage of the Series.

    The memory usage can optionally include the contribution of
    the index and of elements of `object` dtype.

    Parameters
    ----------
    index : bool, default True
        Specifies whether to include the memory usage of the Series index.
    deep : bool, default False
        If True, introspect the data deeply by interrogating
        `object` dtypes for system-level memory consumption, and include
        it in the returned value.

    Returns
    -------
    int
        Bytes of memory consumed.

    See Also
    --------
    numpy.ndarray.nbytes : Total bytes consumed by the elements of the
        array.
    DataFrame.memory_usage : Bytes consumed by a DataFrame.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s = md.Series(range(3))
    >>> s.memory_usage().execute()
    152

    Not including the index gives the size of the rest of the data, which
    is necessarily smaller:

    >>> s.memory_usage(index=False).execute()
    24

    The memory footprint of `object` values is ignored by default:

    >>> s = md.Series(["a", "b"])
    >>> s.values.execute()
    array(['a', 'b'], dtype=object)

    >>> s.memory_usage().execute()
    144

    >>> s.memory_usage(deep=True).execute()
    260
    """
    op = DataFrameMemoryUsage(index=index, deep=deep)
    return op(series)


def index_memory_usage(index, deep=False):
    """
    Memory usage of the values.

    Parameters
    ----------
    deep : bool
        Introspect the data deeply, interrogate
        `object` dtypes for system-level memory consumption.

    Returns
    -------
    bytes used

    See Also
    --------
    numpy.ndarray.nbytes

    Notes
    -----
    Memory usage does not include memory consumed by elements that
    are not components of the array if deep=False
    """
    op = DataFrameMemoryUsage(index=False, deep=deep)
    return op(index)
