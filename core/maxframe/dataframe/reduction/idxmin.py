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
from ..utils import validate_axis
from .core import DataFrameReduction, DataFrameReductionMixin, ReductionCallable


class DataFrameIdxMin(DataFrameReduction, DataFrameReductionMixin):
    _op_type_ = opcodes.IDXMIN
    _func_name = "idxmin"

    @property
    def is_atomic(self):
        return True

    def get_reduction_args(self, axis=None):
        args = dict(skipna=self.skipna)
        if self.inputs and self.inputs[0].ndim > 1:
            args["axis"] = axis
        return {k: v for k, v in args.items() if v is not None}

    @classmethod
    def get_reduction_callable(cls, op):
        func_name = getattr(op, "_func_name")
        kw = dict(skipna=op.skipna)
        kw = {k: v for k, v in kw.items() if v is not None}
        return ReductionCallable(func_name=func_name, kwargs=kw)


def idxmin_dataframe(df, axis=0, skipna=True):
    """
    Return index of first occurrence of minimum over requested axis.

    NA/null values are excluded.

    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
    skipna : bool, default True
        Exclude NA/null values. If an entire row/column is NA, the result
        will be NA.

    Returns
    -------
    Series
        Indexes of minima along the specified axis.

    Raises
    ------
    ValueError
        * If the row/column is empty

    See Also
    --------
    Series.idxmin : Return index of the minimum element.

    Notes
    -----
    This method is the DataFrame version of ``ndarray.argmin``.

    Examples
    --------
    Consider a dataset containing food consumption in Argentina.

    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'consumption': [10.51, 103.11, 55.48],
    ...                    'co2_emissions': [37.2, 19.66, 1712]},
    ...                    index=['Pork', 'Wheat Products', 'Beef'])

    >>> df.execute()
                    consumption  co2_emissions
    Pork                  10.51         37.20
    Wheat Products       103.11         19.66
    Beef                  55.48       1712.00

    By default, it returns the index for the minimum value in each column.

    >>> df.idxmin().execute()
    consumption                Pork
    co2_emissions    Wheat Products
    dtype: object

    To return the index for the minimum value in each row, use ``axis="columns"``.

    >>> df.idxmin(axis="columns").execute()
    Pork                consumption
    Wheat Products    co2_emissions
    Beef                consumption
    dtype: object
    """
    axis = validate_axis(axis, df)
    op = DataFrameIdxMin(
        axis=axis,
        skipna=skipna,
        output_types=[OutputType.series],
    )
    return op(df)


def idxmin_series(series, axis=0, skipna=True):
    """
    Return the row label of the minimum value.

    If multiple values equal the minimum, the first row label with that
    value is returned.

    Parameters
    ----------
    axis : int, default 0
        For compatibility with DataFrame.idxmin. Redundant for application
        on Series.
    skipna : bool, default True
        Exclude NA/null values. If the entire Series is NA, the result
        will be NA.
    *args, **kwargs
        Additional arguments and keywords have no effect but might be
        accepted for compatibility with NumPy.

    Returns
    -------
    Index
        Label of the minimum value.

    Raises
    ------
    ValueError
        If the Series is empty.

    See Also
    --------
    numpy.argmin : Return indices of the minimum values
        along the given axis.
    DataFrame.idxmin : Return index of first occurrence of minimum
        over requested axis.
    Series.idxmin : Return index *label* of the first occurrence
        of minimum of values.

    Notes
    -----
    This method is the Series version of ``ndarray.argmin``. This method
    returns the label of the minimum, while ``ndarray.argmin`` returns
    the position. To get the position, use ``series.values.argmin()``.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s = md.Series(data=[1, None, 4, 3, 4],
    ...               index=['A', 'B', 'C', 'D', 'E'])
    >>> s.execute()
    A    1.0
    B    NaN
    C    4.0
    D    3.0
    E    4.0
    dtype: float64

    >>> s.idxmin().execute()
    'C'

    If `skipna` is False and there is an NA value in the data,
    the function returns ``nan``.

    >>> s.idxmin(skipna=False).execute()
    nan
    """
    validate_axis(axis, series)
    op = DataFrameIdxMin(
        dropna=skipna,
        output_types=[OutputType.scalar],
    )
    return op(series)
