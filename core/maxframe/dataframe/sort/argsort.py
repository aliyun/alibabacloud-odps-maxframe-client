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


def series_argsort(series, axis=0, kind="quicksort", order=None, stable=None):
    """
    Return the integer indices that would sort the Series values.

    Override ndarray.argsort. Argsorts the value, omitting NA/null values,
    and places the result in the same locations as the non-NA values.

    Parameters
    ----------
    axis : {0 or 'index'}
        Unused. Parameter needed for compatibility with DataFrame.
    kind : {'mergesort', 'quicksort', 'heapsort', 'stable'}, default 'quicksort'
        Choice of sorting algorithm. See :func:`numpy.sort` for more
        information. 'mergesort' and 'stable' are the only stable algorithms.
    order : None
        Has no effect but is accepted for compatibility with numpy.
    stable : None
        Has no effect but is accepted for compatibility with numpy.

    Returns
    -------
    Series[np.intp]
        Positions of values within the sort order with -1 indicating
        nan values.

    See Also
    --------
    maxframe.tensor.argsort : Returns the indices that would sort this array.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> s = md.Series([3, 2, 1])
    >>> s.argsort().execute()
    0    2
    1    1
    2    0
    dtype: int64
    """
    from ... import tensor as mt
    from ..datasource.from_tensor import series_from_tensor

    _ = axis, order, stable
    axis = 0
    t = mt.argsort(series.to_tensor(), axis=axis, kind=kind)
    return series_from_tensor(t, index=series.index)


def index_argsort(index, *args, **kwargs):
    from ... import tensor as mt

    return mt.argsort(index.to_tensor(), *args, **kwargs)
