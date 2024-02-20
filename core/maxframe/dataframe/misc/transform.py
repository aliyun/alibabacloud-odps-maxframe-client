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

from typing import Any, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import AnyField, BoolField, DictField, TupleField
from ...utils import pd_release_version, quiet_stdio
from ..core import DATAFRAME_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import (
    build_df,
    build_series,
    make_dtypes,
    pack_func_args,
    parse_index,
    validate_axis,
)

_with_convert_dtype = pd_release_version < (1, 2, 0)


class TransformOperator(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.TRANSFORM

    func = AnyField("func", default=None)
    axis = AnyField("axis", default=None)
    convert_dtype = BoolField("convert_dtype", default=None)
    args = TupleField("args", default=())
    kwds = DictField("kwds", default_factory=dict)

    call_agg = BoolField("call_agg", default=None)

    def __init__(self, output_types=None, memory_scale=None, **kw):
        super().__init__(_output_types=output_types, _memory_scale=memory_scale, **kw)

    def _infer_df_func_returns(self, df, dtypes):
        packed_funcs = self.get_packed_funcs(df)
        test_df = self._build_stub_pandas_obj(df)
        if self.output_types[0] == OutputType.dataframe:
            try:
                with np.errstate(all="ignore"), quiet_stdio():
                    if self.call_agg:
                        infer_df = test_df.agg(packed_funcs, axis=self.axis)
                    else:
                        infer_df = test_df.transform(packed_funcs, axis=self.axis)
            except:  # noqa: E722
                infer_df = None
        else:
            try:
                with np.errstate(all="ignore"), quiet_stdio():
                    if self.call_agg:
                        infer_df = test_df.agg(packed_funcs)
                    else:
                        if not _with_convert_dtype:
                            infer_df = test_df.transform(packed_funcs)
                        else:  # pragma: no cover
                            infer_df = test_df.transform(
                                packed_funcs, convert_dtype=self.convert_dtype
                            )
            except:  # noqa: E722
                infer_df = None

        if infer_df is None and dtypes is None:
            raise TypeError(
                "Failed to infer dtype, please specify dtypes as arguments."
            )

        if infer_df is None:
            is_df = self.output_types[0] == OutputType.dataframe
        else:
            is_df = isinstance(infer_df, pd.DataFrame)

        if is_df:
            new_dtypes = make_dtypes(dtypes) if dtypes is not None else infer_df.dtypes
            self.output_types = [OutputType.dataframe]
        else:
            new_dtypes = (
                dtypes if dtypes is not None else (infer_df.name, infer_df.dtype)
            )
            self.output_types = [OutputType.series]

        return new_dtypes

    def __call__(self, df, dtypes=None, index=None, skip_infer=None):
        axis = getattr(self, "axis", None) or 0
        self.axis = validate_axis(axis, df)
        if not skip_infer:
            dtypes = self._infer_df_func_returns(df, dtypes)

        if self.output_types[0] == OutputType.dataframe:
            new_shape = list(df.shape)
            new_index_value = df.index_value
            if len(new_shape) == 1:
                new_shape.append(len(dtypes) if dtypes is not None else np.nan)
            else:
                new_shape[1] = len(dtypes) if dtypes is not None else np.nan

            if self.call_agg:
                new_shape[self.axis] = np.nan
                new_index_value = parse_index(None, (df.key, df.index_value.key))
            if dtypes is None:
                columns_value = None
            else:
                columns_value = parse_index(dtypes.index, store_data=True)
            return self.new_dataframe(
                [df],
                shape=tuple(new_shape),
                dtypes=dtypes,
                index_value=new_index_value,
                columns_value=columns_value,
            )
        else:
            if dtypes is not None:
                name, dtype = dtypes
            else:
                name, dtype = None, None

            if isinstance(df, DATAFRAME_TYPE):
                new_shape = (df.shape[1 - axis],)
                new_index_value = [df.columns_value, df.index_value][axis]
            else:
                new_shape = (np.nan,) if self.call_agg else df.shape
                new_index_value = df.index_value

            return self.new_series(
                [df],
                shape=new_shape,
                name=name,
                dtype=dtype,
                index_value=new_index_value,
            )

    def get_packed_funcs(self, df=None) -> Any:
        stub_df = self._build_stub_pandas_obj(df or self.inputs[0])
        return pack_func_args(stub_df, self.func, *self.args, **self.kwds)

    def _build_stub_pandas_obj(self, df) -> Union[DataFrame, Series]:
        # TODO: Simulate a dataframe with the corresponding indexes if self.func is
        # a dict and axis=1
        if self.output_types[0] == OutputType.dataframe:
            return build_df(df, fill_value=1, size=1)
        return build_series(df, size=1, name=df.name)


def df_transform(df, func, axis=0, *args, dtypes=None, skip_infer=False, **kwargs):
    """
    Call ``func`` on self producing a DataFrame with transformed values.

    Produced DataFrame will have same axis length as self.

    Parameters
    ----------
    func : function, str, list or dict
        Function to use for transforming the data. If a function, must either
        work when passed a DataFrame or when passed to DataFrame.apply.

        Accepted combinations are:

        - function
        - string function name
        - list of functions and/or function names, e.g. ``[np.exp. 'sqrt']``
        - dict of axis labels -> functions, function names or list of such.
    axis : {0 or 'index', 1 or 'columns'}, default 0
            If 0 or 'index': apply function to each column.
            If 1 or 'columns': apply function to each row.

    dtypes : Series, default None
        Specify dtypes of returned DataFrames. See `Notes` for more details.

    skip_infer: bool, default False
        Whether infer dtypes when dtypes or output_type is not specified.

    *args
        Positional arguments to pass to `func`.
    **kwargs
        Keyword arguments to pass to `func`.

    Returns
    -------
    DataFrame
        A DataFrame that must have the same length as self.

    Raises
    ------
    ValueError : If the returned DataFrame has a different length than self.

    See Also
    --------
    DataFrame.agg : Only perform aggregating type operations.
    DataFrame.apply : Invoke function on a DataFrame.

    Notes
    -----
    When deciding output dtypes and shape of the return value, MaxFrame will
    try applying ``func`` onto a mock DataFrame and the apply call may
    fail. When this happens, you need to specify a list or a pandas
    Series as ``dtypes`` of output DataFrame.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'A': range(3), 'B': range(1, 4)})
    >>> df.execute()
       A  B
    0  0  1
    1  1  2
    2  2  3
    >>> df.transform(lambda x: x + 1).execute()
       A  B
    0  1  2
    1  2  3
    2  3  4

    Even though the resulting DataFrame must have the same length as the
    input DataFrame, it is possible to provide several input functions:

    >>> s = md.Series(range(3))
    >>> s.execute()
    0    0
    1    1
    2    2
    dtype: int64
    >>> s.transform([mt.sqrt, mt.exp]).execute()
           sqrt        exp
    0  0.000000   1.000000
    1  1.000000   2.718282
    2  1.414214   7.389056
    """
    op = TransformOperator(
        func=func,
        axis=axis,
        args=args,
        kwds=kwargs,
        output_types=[OutputType.dataframe],
        call_agg=kwargs.pop("_call_agg", False),
    )
    return op(df, dtypes=dtypes, skip_infer=skip_infer)


def series_transform(
    series,
    func,
    convert_dtype=True,
    axis=0,
    *args,
    skip_infer=False,
    dtype=None,
    **kwargs
):
    """
    Call ``func`` on self producing a Series with transformed values.

    Produced Series will have same axis length as self.

    Parameters
    ----------
    func : function, str, list or dict
    Function to use for transforming the data. If a function, must either
    work when passed a Series or when passed to Series.apply.

    Accepted combinations are:

    - function
    - string function name
    - list of functions and/or function names, e.g. ``[np.exp. 'sqrt']``
    - dict of axis labels -> functions, function names or list of such.
    axis : {0 or 'index'}
        Parameter needed for compatibility with DataFrame.

    dtype : numpy.dtype, default None
        Specify dtypes of returned DataFrames. See `Notes` for more details.

    skip_infer: bool, default False
        Whether infer dtypes when dtypes or output_type is not specified.

    *args
        Positional arguments to pass to `func`.
    **kwargs
        Keyword arguments to pass to `func`.

    Returns
    -------
    Series
    A Series that must have the same length as self.

    Raises
    ------
    ValueError : If the returned Series has a different length than self.

    See Also
    --------
    Series.agg : Only perform aggregating type operations.
    Series.apply : Invoke function on a Series.

    Notes
    -----
    When deciding output dtypes and shape of the return value, MaxFrame will
    try applying ``func`` onto a mock Series, and the transform call may
    fail. When this happens, you need to specify ``dtype`` of output
    Series.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'A': range(3), 'B': range(1, 4)})
    >>> df.execute()
    A  B
    0  0  1
    1  1  2
    2  2  3
    >>> df.transform(lambda x: x + 1).execute()
    A  B
    0  1  2
    1  2  3
    2  3  4

    Even though the resulting Series must have the same length as the
    input Series, it is possible to provide several input functions:

    >>> s = md.Series(range(3))
    >>> s.execute()
    0    0
    1    1
    2    2
    dtype: int64
    >>> s.transform([mt.sqrt, mt.exp]).execute()
       sqrt        exp
    0  0.000000   1.000000
    1  1.000000   2.718282
    2  1.414214   7.389056
    """
    op = TransformOperator(
        func=func,
        axis=axis,
        convert_dtype=convert_dtype,
        args=args,
        kwds=kwargs,
        output_types=[OutputType.series],
        call_agg=kwargs.pop("_call_agg", False),
    )
    dtypes = (series.name, dtype) if dtype is not None else None
    return op(series, dtypes=dtypes, skip_infer=skip_infer)
