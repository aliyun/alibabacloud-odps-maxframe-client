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

import inspect
from typing import Any, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from ... import opcodes
from ...core import OutputType
from ...core.operator import OperatorLogicKeyGeneratorMixin
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    FunctionField,
    StringField,
    TupleField,
)
from ...utils import get_func_token, quiet_stdio, tokenize
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import (
    build_df,
    build_series,
    make_dtype,
    make_dtypes,
    pack_func_args,
    parse_index,
    validate_axis,
    validate_output_types,
)


class ApplyOperandLogicKeyGeneratorMixin(OperatorLogicKeyGeneratorMixin):
    def _get_logic_key_token_values(self):
        token_values = super()._get_logic_key_token_values() + [
            self.axis,
            self.convert_dtype,
            self.raw,
            self.result_type,
            self.elementwise,
        ]
        if self.func:
            return token_values + [get_func_token(self.func)]
        else:  # pragma: no cover
            return token_values


class ApplyOperator(
    DataFrameOperator, DataFrameOperatorMixin, ApplyOperandLogicKeyGeneratorMixin
):
    _op_type_ = opcodes.APPLY

    func = FunctionField("func")
    axis = AnyField("axis", default=0)
    convert_dtype = BoolField("convert_dtype", default=True)
    raw = BoolField("raw", default=False)
    result_type = StringField("result_type", default=None)
    elementwise = BoolField("elementwise")
    logic_key = StringField("logic_key")
    need_clean_up_func = BoolField("need_clean_up_func")
    args = TupleField("args", default=())
    kwds = DictField("kwds", default={})

    def __init__(self, output_type=None, **kw):
        if output_type:
            kw["_output_types"] = [output_type]
        super().__init__(**kw)

    def _update_key(self):
        values = [v for v in self._values_ if v is not self.func] + [
            get_func_token(self.func)
        ]
        self._obj_set("_key", tokenize(type(self).__name__, *values))
        return self

    def _infer_df_func_returns(self, df, dtypes, dtype=None, name=None, index=None):
        if isinstance(self.func, np.ufunc):
            output_type = OutputType.dataframe
            new_dtypes = None
            index_value = "inherit"
            new_elementwise = True
        else:
            if self.output_types is not None and (
                dtypes is not None or dtype is not None
            ):
                ret_dtypes = dtypes if dtypes is not None else (name, dtype)
                ret_index_value = parse_index(index) if index is not None else None
                self.elementwise = False
                return ret_dtypes, ret_index_value

            output_type = new_dtypes = index_value = None
            new_elementwise = False

        try:
            empty_df = build_df(df, size=2)
            with np.errstate(all="ignore"), quiet_stdio():
                infer_df = empty_df.apply(
                    self.func,
                    axis=self.axis,
                    raw=self.raw,
                    result_type=self.result_type,
                    args=self.args,
                    **self.kwds,
                )
            if index_value is None:
                if infer_df.index is empty_df.index:
                    index_value = "inherit"
                else:
                    index_value = parse_index(pd.RangeIndex(-1))

            if isinstance(infer_df, pd.DataFrame):
                output_type = output_type or OutputType.dataframe
                new_dtypes = new_dtypes or infer_df.dtypes
            else:
                output_type = output_type or OutputType.series
                new_dtypes = (name or infer_df.name, dtype or infer_df.dtype)
            new_elementwise = False if new_elementwise is None else new_elementwise
        except:  # noqa: E722  # nosec
            pass

        self.output_types = (
            [output_type]
            if not self.output_types and output_type
            else self.output_types
        )
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        self.elementwise = (
            new_elementwise if self.elementwise is None else self.elementwise
        )
        return dtypes, index_value

    def _call_df_or_series(self, df):
        return self.new_df_or_series([df])

    def _call_dataframe(self, df, dtypes=None, dtype=None, name=None, index=None):
        # for backward compatibility
        dtype = dtype if dtype is not None else dtypes
        dtypes, index_value = self._infer_df_func_returns(
            df, dtypes, dtype=dtype, name=name, index=index
        )
        if index_value is None:
            index_value = parse_index(None, (df.key, df.index_value.key))
        for arg, desc in zip((self.output_types, dtypes), ("output_types", "dtypes")):
            if arg is None:
                raise TypeError(
                    f"Cannot determine {desc} by calculating with enumerate data, "
                    "please specify it as arguments"
                )

        if index_value == "inherit":
            index_value = df.index_value

        if self.elementwise:
            shape = df.shape
        elif self.output_types[0] == OutputType.dataframe:
            shape = [np.nan, np.nan]
            shape[1 - self.axis] = df.shape[1 - self.axis]
            shape = tuple(shape)
        else:
            shape = (df.shape[1 - self.axis],)

        if self.output_types[0] == OutputType.dataframe:
            if self.axis == 0:
                return self.new_dataframe(
                    [df],
                    shape=shape,
                    dtypes=dtypes,
                    index_value=index_value,
                    columns_value=parse_index(dtypes.index, store_data=True),
                )
            else:
                return self.new_dataframe(
                    [df],
                    shape=shape,
                    dtypes=dtypes,
                    index_value=df.index_value,
                    columns_value=parse_index(dtypes.index, store_data=True),
                )
        else:
            name, dtype = dtypes
            return self.new_series(
                [df], shape=shape, name=name, dtype=dtype, index_value=index_value
            )

    def _call_series(self, series, dtypes=None, dtype=None, name=None, index=None):
        # for backward compatibility
        dtype = dtype if dtype is not None else dtypes
        if self.convert_dtype:
            if self.output_types is not None and (
                dtypes is not None or dtype is not None
            ):
                infer_series = test_series = None
            else:
                test_series = build_series(series, size=2, name=series.name)
                try:
                    with np.errstate(all="ignore"), quiet_stdio():
                        infer_series = test_series.apply(
                            self.func, args=self.args, **self.kwds
                        )
                except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                    infer_series = None

            output_type = self._output_types[0]

            if index is not None:
                index_value = parse_index(index)
            elif infer_series is not None:
                if infer_series.index is test_series.index:
                    index_value = series.index_value
                else:  # pragma: no cover
                    index_value = parse_index(infer_series.index)
            else:
                index_value = parse_index(None, series)

            if output_type == OutputType.dataframe:
                if dtypes is None:
                    if infer_series is not None and infer_series.ndim == 2:
                        dtypes = infer_series.dtypes
                    else:
                        raise TypeError(
                            "Cannot determine dtypes, "
                            "please specify `dtypes` as argument"
                        )
                columns_value = parse_index(dtypes.index, store_data=True)

                return self.new_dataframe(
                    [series],
                    shape=(series.shape[0], len(dtypes)),
                    index_value=index_value,
                    columns_value=columns_value,
                    dtypes=dtypes,
                )
            else:
                if (
                    dtype is None
                    and infer_series is not None
                    and infer_series.ndim == 1
                ):
                    dtype = infer_series.dtype
                else:
                    dtype = dtype if dtype is not None else np.dtype(object)
                if infer_series is not None and infer_series.ndim == 1:
                    name = name or infer_series.name
                return self.new_series(
                    [series],
                    dtype=dtype,
                    shape=series.shape,
                    index_value=index_value,
                    name=name,
                )
        else:
            dtype = dtype if dtype is not None else np.dtype("object")
            return self.new_series(
                [series],
                dtype=dtype,
                shape=series.shape,
                index_value=series.index_value,
                name=name,
            )

    def __call__(self, df_or_series, dtypes=None, dtype=None, name=None, index=None):
        axis = getattr(self, "axis", None) or 0
        dtypes = make_dtypes(dtypes)
        dtype = make_dtype(dtype)
        self.axis = validate_axis(axis, df_or_series)

        if self.output_types and self.output_types[0] == OutputType.df_or_series:
            return self._call_df_or_series(df_or_series)

        if df_or_series.op.output_types[0] == OutputType.dataframe:
            return self._call_dataframe(
                df_or_series, dtypes=dtypes, dtype=dtype, name=name, index=index
            )
        else:
            return self._call_series(
                df_or_series, dtypes=dtypes, dtype=dtype, name=name, index=index
            )

    def _build_stub_pandas_obj(self, df_or_series) -> Union[DataFrame, Series]:
        if self.output_types[0] == OutputType.dataframe:
            return build_df(df_or_series, size=2)
        return build_series(df_or_series, size=2, name=df_or_series.name)

    def get_packed_funcs(self, df=None) -> Any:
        stub_df = self._build_stub_pandas_obj(df or self.inputs[0])
        return pack_func_args(stub_df, self.func, *self.args, **self.kwds)


def df_apply(
    df,
    func,
    axis=0,
    raw=False,
    result_type=None,
    args=(),
    dtypes=None,
    dtype=None,
    name=None,
    output_type=None,
    index=None,
    elementwise=None,
    skip_infer=False,
    **kwds,
):
    """
    Apply a function along an axis of the DataFrame.

    Objects passed to the function are Series objects whose index is
    either the DataFrame's index (``axis=0``) or the DataFrame's columns
    (``axis=1``). By default (``result_type=None``), the final return type
    is inferred from the return type of the applied function. Otherwise,
    it depends on the `result_type` argument.

    Parameters
    ----------
    func : function
        Function to apply to each column or row.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Axis along which the function is applied:

        * 0 or 'index': apply function to each column.
        * 1 or 'columns': apply function to each row.

    raw : bool, default False
        Determines if row or column is passed as a Series or ndarray object:

        * ``False`` : passes each row or column as a Series to the
          function.
        * ``True`` : the passed function will receive ndarray objects
          instead.
          If you are just applying a NumPy reduction function this will
          achieve much better performance.

    result_type : {'expand', 'reduce', 'broadcast', None}, default None
        These only act when ``axis=1`` (columns):

        * 'expand' : list-like results will be turned into columns.
        * 'reduce' : returns a Series if possible rather than expanding
          list-like results. This is the opposite of 'expand'.
        * 'broadcast' : results will be broadcast to the original shape
          of the DataFrame, the original index and columns will be
          retained.

        The default behaviour (None) depends on the return value of the
        applied function: list-like results will be returned as a Series
        of those. However if the apply function returns a Series these
        are expanded to columns.

    output_type : {'dataframe', 'series'}, default None
        Specify type of returned object. See `Notes` for more details.

    dtypes : Series, default None
        Specify dtypes of returned DataFrames. See `Notes` for more details.

    dtype : numpy.dtype, default None
        Specify dtype of returned Series. See `Notes` for more details.

    name : str, default None
        Specify name of returned Series. See `Notes` for more details.

    index : Index, default None
        Specify index of returned object. See `Notes` for more details.

    elementwise : bool, default False
        Specify whether ``func`` is an elementwise function:

        * ``False`` : The function is not elementwise. MaxFrame will try
          concatenating chunks in rows (when ``axis=0``) or in columns
          (when ``axis=1``) and then apply ``func`` onto the concatenated
          chunk. The concatenation step can cause extra latency.
        * ``True`` : The function is elementwise. MaxFrame will apply
          ``func`` to original chunks. This will not introduce extra
          concatenation step and reduce overhead.

    skip_infer: bool, default False
        Whether infer dtypes when dtypes or output_type is not specified.

    args : tuple
        Positional arguments to pass to `func` in addition to the
        array/series.

    **kwds
        Additional keyword arguments to pass as keywords arguments to
        `func`.

    Returns
    -------
    Series or DataFrame
        Result of applying ``func`` along the given axis of the
        DataFrame.

    See Also
    --------
    DataFrame.applymap: For elementwise operations.
    DataFrame.aggregate: Only perform aggregating type operations.
    DataFrame.transform: Only perform transforming type operations.

    Notes
    -----
    When deciding output dtypes and shape of the return value, MaxFrame will
    try applying ``func`` onto a mock DataFrame,  and the apply call may
    fail. When this happens, you need to specify the type of apply call
    (DataFrame or Series) in output_type.

    * For DataFrame output, you need to specify a list or a pandas Series
      as ``dtypes`` of output DataFrame. ``index`` of output can also be
      specified.
    * For Series output, you need to specify ``dtype`` and ``name`` of
      output Series.

    Examples
    --------
    >>> import numpy as np
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
    >>> df.execute()
       A  B
    0  4  9
    1  4  9
    2  4  9

    Using a reducing function on either axis

    >>> df.apply(np.sum, axis=0).execute()
    A    12
    B    27
    dtype: int64

    >>> df.apply(np.sum, axis=1).execute()
    0    13
    1    13
    2    13
    dtype: int64

    Returning a list-like will result in a Series

    >>> df.apply(lambda x: [1, 2], axis=1).execute()
    0    [1, 2]
    1    [1, 2]
    2    [1, 2]
    dtype: object

    Passing ``result_type='expand'`` will expand list-like results
    to columns of a Dataframe

    >>> df.apply(lambda x: [1, 2], axis=1, result_type='expand').execute()
       0  1
    0  1  2
    1  1  2
    2  1  2

    Returning a Series inside the function is similar to passing
    ``result_type='expand'``. The resulting column names
    will be the Series index.

    >>> df.apply(lambda x: md.Series([1, 2], index=['foo', 'bar']), axis=1).execute()
       foo  bar
    0    1    2
    1    1    2
    2    1    2

    Passing ``result_type='broadcast'`` will ensure the same shape
    result, whether list-like or scalar is returned by the function,
    and broadcast it along the axis. The resulting column names will
    be the originals.

    >>> df.apply(lambda x: [1, 2], axis=1, result_type='broadcast').execute()
       A  B
    0  1  2
    1  1  2
    2  1  2
    """
    if isinstance(func, (list, dict)):
        return df.aggregate(func, axis)

    output_types = kwds.pop("output_types", None)
    object_type = kwds.pop("object_type", None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type
    )
    output_type = output_types[0] if output_types else None
    if skip_infer and output_type is None:
        output_type = OutputType.df_or_series

    # calling member function
    if isinstance(func, str):
        func = getattr(df, func)
        sig = inspect.getfullargspec(func)
        if "axis" in sig.args:
            kwds["axis"] = axis
        return func(*args, **kwds)

    op = ApplyOperator(
        func=func,
        axis=axis,
        raw=raw,
        result_type=result_type,
        args=args,
        kwds=kwds,
        output_type=output_type,
        elementwise=elementwise,
    )
    return op(df, dtypes=dtypes, dtype=dtype, name=name, index=index)


def series_apply(
    series,
    func,
    convert_dtype=True,
    output_type=None,
    args=(),
    dtypes=None,
    dtype=None,
    name=None,
    index=None,
    skip_infer=False,
    **kwds,
):
    """
    Invoke function on values of Series.

    Can be ufunc (a NumPy function that applies to the entire Series)
    or a Python function that only works on single values.

    Parameters
    ----------
    func : function
        Python function or NumPy ufunc to apply.

    convert_dtype : bool, default True
        Try to find better dtype for elementwise function results. If
        False, leave as dtype=object.

    output_type : {'dataframe', 'series'}, default None
        Specify type of returned object. See `Notes` for more details.

    dtypes : Series, default None
        Specify dtypes of returned DataFrames. See `Notes` for more details.

    dtype : numpy.dtype, default None
        Specify dtype of returned Series. See `Notes` for more details.

    name : str, default None
        Specify name of returned Series. See `Notes` for more details.

    index : Index, default None
        Specify index of returned object. See `Notes` for more details.

    args : tuple
        Positional arguments passed to func after the series value.

    skip_infer: bool, default False
        Whether infer dtypes when dtypes or output_type is not specified.

    **kwds
        Additional keyword arguments passed to func.

    Returns
    -------
    Series or DataFrame
        If func returns a Series object the result will be a DataFrame.

    See Also
    --------
    Series.map: For element-wise operations.
    Series.agg: Only perform aggregating type operations.
    Series.transform: Only perform transforming type operations.

    Notes
    -----
    When deciding output dtypes and shape of the return value, MaxFrame will
    try applying ``func`` onto a mock Series, and the apply call may fail.
    When this happens, you need to specify the type of apply call
    (DataFrame or Series) in output_type.

    * For DataFrame output, you need to specify a list or a pandas Series
      as ``dtypes`` of output DataFrame. ``index`` of output can also be
      specified.
    * For Series output, you need to specify ``dtype`` and ``name`` of
      output Series.

    Examples
    --------
    Create a series with typical summer temperatures for each city.

    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> s = md.Series([20, 21, 12],
    ...               index=['London', 'New York', 'Helsinki'])
    >>> s.execute()
    London      20
    New York    21
    Helsinki    12
    dtype: int64

    Square the values by defining a function and passing it as an
    argument to ``apply()``.

    >>> def square(x):
    ...     return x ** 2
    >>> s.apply(square).execute()
    London      400
    New York    441
    Helsinki    144
    dtype: int64

    Square the values by passing an anonymous function as an
    argument to ``apply()``.

    >>> s.apply(lambda x: x ** 2).execute()
    London      400
    New York    441
    Helsinki    144
    dtype: int64

    Define a custom function that needs additional positional
    arguments and pass these additional arguments using the
    ``args`` keyword.

    >>> def subtract_custom_value(x, custom_value):
    ...     return x - custom_value

    >>> s.apply(subtract_custom_value, args=(5,)).execute()
    London      15
    New York    16
    Helsinki     7
    dtype: int64

    Define a custom function that takes keyword arguments
    and pass these arguments to ``apply``.

    >>> def add_custom_values(x, **kwargs):
    ...     for month in kwargs:
    ...         x += kwargs[month]
    ...     return x

    >>> s.apply(add_custom_values, june=30, july=20, august=25).execute()
    London      95
    New York    96
    Helsinki    87
    dtype: int64
    """
    if isinstance(func, (list, dict)):
        return series.aggregate(func)

    # calling member function
    if isinstance(func, str):
        func_body = getattr(series, func, None)
        if func_body is not None:
            return func_body(*args, **kwds)
        func_str = func
        func = getattr(np, func_str, None)
        if func is None:
            raise AttributeError(
                f"'{func_str!r}' is not a valid function "
                f"for '{type(series).__name__}' object"
            )

    if skip_infer and output_type is None:
        output_type = OutputType.df_or_series

    output_types = kwds.pop("output_types", None)
    object_type = kwds.pop("object_type", None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type
    )
    output_type = output_types[0] if output_types else OutputType.series

    op = ApplyOperator(
        func=func,
        convert_dtype=convert_dtype,
        args=args,
        kwds=kwds,
        output_type=output_type,
    )
    return op(series, dtypes=dtypes, dtype=dtype, name=name, index=index)
