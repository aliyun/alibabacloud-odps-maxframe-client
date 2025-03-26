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

import functools
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import (
    DictField,
    FunctionField,
    Int32Field,
    TupleField,
)
from ...utils import quiet_stdio
from ..core import DATAFRAME_TYPE, DataFrame, IndexValue, Series
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import (
    build_df,
    build_series,
    copy_func_scheduling_hints,
    make_dtypes,
    pack_func_args,
    parse_index,
    validate_output_types,
)


class DataFrameApplyChunkOperator(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.APPLY_CHUNK

    func = FunctionField("func")
    batch_rows = Int32Field("batch_rows", default=None)
    args = TupleField("args", default=None)
    kwargs = DictField("kwargs", default=None)

    def __init__(self, output_type=None, **kw):
        if output_type:
            kw["_output_types"] = [output_type]
        super().__init__(**kw)
        if hasattr(self, "func"):
            copy_func_scheduling_hints(self.func, self)

    def _call_dataframe(self, df, dtypes, index_value, element_wise):
        # return dataframe
        if self.output_types[0] == OutputType.dataframe:
            dtypes = make_dtypes(dtypes)
            # apply_chunk will use generate new range index for results
            return self.new_dataframe(
                [df],
                shape=df.shape if element_wise else (np.nan, len(dtypes)),
                index_value=index_value,
                columns_value=parse_index(dtypes.index, store_data=True),
                dtypes=dtypes,
            )

        # return series
        if not isinstance(dtypes, tuple):
            raise TypeError(
                "Cannot determine dtype, " "please specify `dtype` as argument"
            )

        name, dtype = dtypes
        return self.new_series(
            [df], shape=(np.nan,), name=name, dtype=dtype, index_value=index_value
        )

    def _call_series(self, series, dtypes, index_value, element_wise):
        if self.output_types[0] == OutputType.series:
            if not isinstance(dtypes, tuple):
                raise TypeError(
                    "Cannot determine dtype, " "please specify `dtype` as argument"
                )

            name, dtype = dtypes
            shape = series.shape if element_wise else (np.nan,)

            return self.new_series(
                [series],
                dtype=dtype,
                shape=shape,
                index_value=index_value,
                name=name,
            )

        dtypes = make_dtypes(dtypes)
        return self.new_dataframe(
            [series],
            shape=(np.nan, len(dtypes)),
            index_value=index_value,
            columns_value=parse_index(dtypes.index, store_data=True),
            dtypes=dtypes,
        )

    def __call__(
        self,
        df_or_series: Union[DataFrame, Series],
        dtypes: Union[Tuple[str, Any], Dict[str, Any]] = None,
        output_type=None,
        index=None,
    ):
        args = self.args or ()
        kwargs = self.kwargs or {}
        # if not dtypes and not skip_infer:
        packed_func = get_packed_func(df_or_series, self.func, *args, **kwargs)

        # if skip_infer, directly build a frame
        if self.output_types and self.output_types[0] == OutputType.df_or_series:
            return self.new_df_or_series([df_or_series])

        # infer return index and dtypes
        dtypes, index_value, elementwise = self._infer_batch_func_returns(
            df_or_series,
            origin_func=self.func,
            packed_func=packed_func,
            given_output_type=output_type,
            given_dtypes=dtypes,
            given_index=index,
        )

        if index_value is None:
            index_value = parse_index(
                None, (df_or_series.key, df_or_series.index_value.key, self.func)
            )
        for arg, desc in zip((self.output_types, dtypes), ("output_types", "dtypes")):
            if arg is None:
                raise TypeError(
                    f"Cannot determine {desc} by calculating with enumerate data, "
                    "please specify it as arguments"
                )

        if dtypes is None or len(dtypes) == 0:
            raise TypeError(
                "Cannot determine {dtypes} or {dtype} by calculating with enumerate data, "
                "please specify it as arguments"
            )

        if isinstance(df_or_series, DATAFRAME_TYPE):
            return self._call_dataframe(
                df_or_series,
                dtypes=dtypes,
                index_value=index_value,
                element_wise=elementwise,
            )

        return self._call_series(
            df_or_series,
            dtypes=dtypes,
            index_value=index_value,
            element_wise=elementwise,
        )

    def _infer_batch_func_returns(
        self,
        input_df_or_series: Union[DataFrame, Series],
        origin_func: Union[str, Callable, np.ufunc],
        packed_func: Union[Callable, functools.partial],
        given_output_type: OutputType,
        given_dtypes: Union[Tuple[str, Any], pd.Series, List[Any], Dict[str, Any]],
        given_index: Union[pd.Index, IndexValue],
        given_elementwise: bool = False,
        *args,
        **kwargs,
    ):
        inferred_output_type = inferred_dtypes = inferred_index_value = None
        inferred_is_elementwise = False

        # handle numpy ufunc case
        if isinstance(origin_func, np.ufunc):
            inferred_output_type = OutputType.dataframe
            inferred_dtypes = None
            inferred_index_value = input_df_or_series.index_value
            inferred_is_elementwise = True
        elif self.output_types is not None and given_dtypes is not None:
            inferred_dtypes = given_dtypes

        # build same schema frame toto execute
        if isinstance(input_df_or_series, DATAFRAME_TYPE):
            empty_data = build_df(input_df_or_series, fill_value=1, size=1)
        else:
            empty_data = build_series(
                input_df_or_series, size=1, name=input_df_or_series.name
            )

        try:
            # execute
            with np.errstate(all="ignore"), quiet_stdio():
                infer_result = packed_func(empty_data, *args, **kwargs)

            #  if executed successfully, get index and dtypes from returned object
            if inferred_index_value is None:
                if (
                    infer_result is None
                    or not hasattr(infer_result, "index")
                    or infer_result.index is None
                ):
                    inferred_index_value = parse_index(pd.RangeIndex(-1))
                elif infer_result.index is empty_data.index:
                    inferred_index_value = input_df_or_series.index_value
                else:
                    inferred_index_value = parse_index(infer_result.index, packed_func)

            if isinstance(infer_result, pd.DataFrame):
                if (
                    given_output_type is not None
                    and given_output_type != OutputType.dataframe
                ):
                    raise TypeError(
                        f'Cannot infer output_type as "series", '
                        f'please specify `output_type` as "dataframe"'
                    )
                inferred_output_type = given_output_type or OutputType.dataframe
                inferred_dtypes = (
                    given_dtypes if given_dtypes is not None else infer_result.dtypes
                )
            else:
                if (
                    given_output_type is not None
                    and given_output_type == OutputType.dataframe
                ):
                    raise TypeError(
                        f'Cannot infer output_type as "dataframe", '
                        f'please specify `output_type` as "series"'
                    )
                inferred_output_type = given_output_type or OutputType.series
                inferred_dtypes = (infer_result.name, infer_result.dtype)
        except:  # noqa: E722
            pass

        # merge specified and inferred index, dtypes, output_type
        # elementwise used to decide shape
        self.output_types = (
            [inferred_output_type]
            if not self.output_types and inferred_output_type
            else self.output_types
        )
        inferred_dtypes = given_dtypes if given_dtypes is not None else inferred_dtypes
        if given_index is not None:
            inferred_index_value = (
                parse_index(given_index)
                if given_index is not input_df_or_series.index_value
                else input_df_or_series.index_value
            )
        inferred_is_elementwise = given_elementwise or inferred_is_elementwise
        return inferred_dtypes, inferred_index_value, inferred_is_elementwise


def get_packed_func(df, func, *args, **kwargs) -> Any:
    stub_df = build_df(df, fill_value=1, size=1)
    return pack_func_args(stub_df, func, *args, **kwargs)


def df_apply_chunk(
    dataframe,
    func: Union[str, Callable],
    batch_rows=None,
    dtypes=None,
    dtype=None,
    name=None,
    output_type=None,
    index=None,
    skip_infer=False,
    args=(),
    **kwargs,
):
    """
    Apply a function that takes pandas DataFrame and outputs pandas DataFrame/Series.
    The pandas DataFrame given to the function is a chunk of the input dataframe, consider as a batch rows.

    The objects passed into this function are slices of the original DataFrame, containing at most batch_rows
    number of rows and all columns. It is equivalent to merging multiple ``df.apply`` with ``axis=1`` inputs and then
    passing them into the function for execution, thereby improving performance in specific scenarios. The function
    output can be either a DataFrame or a Series. ``apply_chunk`` will ultimately merge the results into a new
    DataFrame or Series.

    Don't expect to receive all rows of the DataFrame in the function, as it depends on the implementation
    of MaxFrame and the internal running state of MaxCompute.

    Parameters
    ----------
    func : str or Callable
        Function to apply to the dataframe chunk.

    batch_rows : int
        Specify expected number of rows in a batch, as well as the len of function input dataframe. When the remaining
        data is insufficient, it may be less than this number.

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

    skip_infer: bool, default False
        Whether infer dtypes when dtypes or output_type is not specified.

    args : tuple
        Positional arguments to pass to ``func`` in addition to the
        array/series.

    **kwds
        Additional keyword arguments to pass as keywords arguments to
        ``func``.

    Returns
    -------
    Series or DataFrame
        Result of applying ``func`` along the given chunk of the
        DataFrame.

    See Also
    --------
    DataFrame.apply: For non-batching operations.
    Series.mf.apply_chunk: Apply function to Series chunk.

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
    * For any input with data type ``pandas.ArrowDtype(pyarrow.MapType)``, it will always
      be converted to a Python dict. And for any output with this data type, it must be
      returned as a Python dict as well.

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

    Use different batch_rows will collect different dataframe chunk into the function.

    For example, when you use ``batch_rows=3``, it means that the function will wait until 3 rows are collected.

    >>> df.mf.apply_chunk(np.sum, batch_rows=3).execute()
    A    12
    B    27
    dtype: int64

    While, if ``batch_rows=2``, the data will be divided into at least two segments. Additionally, if your function
    alters the shape of the dataframe, it may result in different outputs.

    >>> df.mf.apply_chunk(np.sum, batch_rows=2).execute()
    A     8
    B    18
    A     4
    B     9
    dtype: int64

    If the function requires some parameters, you can specify them using args or kwargs.

    >>> def calc(df, x, y):
    ...    return df * x + y
    >>> df.mf.apply_chunk(calc, args=(10,), y=20).execute()
        A    B
    0  60  110
    1  60  110
    2  60  110

    The batch rows will benefit the actions consume a dataframe, like sklearn predict.
    You can easily use sklearn in MaxFrame to perform offline inference, and apply_chunk makes this process more
    efficient. The ``@with_python_requirements`` provides the capability to automatically package and load
    dependencies.

    Once you rely on some third-party dependencies, MaxFrame may not be able to correctly infer the return type.
    Therefore, using ``output_type`` with ``dtype`` or ``dtypes`` is necessary.

    >>> from maxframe.udf import with_python_requirements
    >>> data = {
    ...     'A': np.random.rand(10),
    ...     'B': np.random.rand(10)
    ... }
    >>> pd_df = pd.DataFrame(data)
    >>> X = pd_df[['A']]
    >>> y = pd_df['B']

    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LinearRegression
    >>> model = LinearRegression()
    >>> model.fit(X, y)

    >>> @with_python_requirements("scikit-learn")
    ... def predict(df):
    ...     predict_B = model.predict(df[["A"]])
    ...     return pd.Series(predict_B, index=df.A.index)

    >>> df.mf.apply_chunk(predict, batch_rows=3, output_type="series", dtype="float", name="predict_B").execute()
    0   -0.765025
    1   -0.765025
    2   -0.765025
    Name: predict_B, dtype: float64

    Create a dataframe with a dict type.

    >>> import pyarrow as pa
    >>> import pandas as pd
    >>> from maxframe.lib.dtypes_extension import dict_
    >>> col_a = pd.Series(
    ...     data=[[("k1", 1), ("k2", 2)], [("k1", 3)], None],
    ...     index=[1, 2, 3],
    ...     dtype=dict_(pa.string(), pa.int64()),
    ... )
    >>> col_b = pd.Series(
    ...     data=["A", "B", "C"],
    ...     index=[1, 2, 3],
    ... )
    >>> df = md.DataFrame({"A": col_a, "B": col_b})
    >>> df.execute()
                            A  B
    1  [('k1', 1), ('k2', 2)]  A
    2             [('k1', 3)]  B
    3                    <NA>  C

    Define a function that updates the map type with a new key-value pair in a batch.

    >>> def custom_set_item(df):
    ...     for name, value in df["A"].items():
    ...         if value is not None:
    ...             df["A"][name]["x"] = 100
    ...     return df

    >>> mf.apply_chunk(
    ...     process,
    ...     output_type="dataframe",
    ...     dtypes=md_df.dtypes.copy(),
    ...     batch_rows=2,
    ...     skip_infer=True,
    ...     index=md_df.index,
    ... )
                                        A  B
    1  [('k1', 1), ('k2', 2), ('x', 10))]  A
    2              [('k1', 3), ('x', 10)]  B
    3                                <NA>  C
    """
    if not isinstance(func, Callable):
        raise TypeError("function must be a callable object")

    if batch_rows is not None:
        if not isinstance(batch_rows, int):
            raise TypeError("batch_rows must be an integer")
        elif batch_rows <= 0:
            raise ValueError("batch_rows must be greater than 0")

    dtypes = (name, dtype) if dtype is not None else dtypes

    output_types = kwargs.pop("output_types", None)
    object_type = kwargs.pop("object_type", None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type
    )
    output_type = output_types[0] if output_types else None
    if skip_infer and output_type is None:
        output_type = OutputType.df_or_series

    # bind args and kwargs
    op = DataFrameApplyChunkOperator(
        func=func,
        batch_rows=batch_rows,
        output_type=output_type,
        args=args,
        kwargs=kwargs,
    )

    return op(
        dataframe,
        dtypes=dtypes,
        index=index,
    )


def series_apply_chunk(
    dataframe_or_series,
    func: Union[str, Callable],
    batch_rows,
    dtypes=None,
    dtype=None,
    name=None,
    output_type=None,
    index=None,
    skip_infer=False,
    args=(),
    **kwargs,
):
    """
    Apply a function that takes pandas Series and outputs pandas DataFrame/Series.
    The pandas DataFrame given to the function is a chunk of the input series.

    The objects passed into this function are slices of the original series, containing at most batch_rows
    number of elements. The function output can be either a DataFrame or a Series.
    ``apply_chunk`` will ultimately merge the results into a new DataFrame or Series.

    Don't expect to receive all elements of series in the function, as it depends on the implementation
    of MaxFrame and the internal running state of MaxCompute.

    Can be ufunc (a NumPy function that applies to the entire Series)
    or a Python function that only works on series.

    Parameters
    ----------
    func : function
        Python function or NumPy ufunc to apply.

    batch_rows : int
        Specify expected number of elements in a batch, as well as the len of function input series.
        When the remaining data is insufficient, it may be less than this number.

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
        If func returns a Series object the result will be a Series, else the result will be a DataFrame.

    See Also
    --------
    DataFrame.apply_chunk: Apply function to DataFrame chunk.
    Series.apply: For non-batching operations.

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
    * For any input with data type ``pandas.ArrowDtype(pyarrow.MapType)``, it will always
      be converted to a Python dict. And for any output with this data type, it must be
      returned as a Python dict as well.

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
    argument to ``apply_chunk()``.

    >>> def square(x):
    ...     return x ** 2
    >>> s.mf.apply_chunk(square, batch_rows=2).execute()
    London      400
    New York    441
    Helsinki    144
    dtype: int64

    Square the values by passing an anonymous function as an
    argument to ``apply_chunk()``.

    >>> s.mf.apply_chunk(lambda x: x**2, batch_rows=2).execute()
    London      400
    New York    441
    Helsinki    144
    dtype: int64

    Define a custom function that needs additional positional
    arguments and pass these additional arguments using the
    ``args`` keyword.

    >>> def subtract_custom_value(x, custom_value):
    ...     return x - custom_value

    >>> s.mf.apply_chunk(subtract_custom_value, args=(5,), batch_rows=3).execute()
    London      15
    New York    16
    Helsinki     7
    dtype: int64

    Define a custom function that takes keyword arguments
    and pass these arguments to ``apply_chunk``.

    >>> def add_custom_values(x, **kwargs):
    ...     for month in kwargs:
    ...         x += kwargs[month]
    ...     return x

    >>> s.mf.apply_chunk(add_custom_values, batch_rows=2, june=30, july=20, august=25).execute()
    London      95
    New York    96
    Helsinki    87
    dtype: int64

    If func return a dataframe, the apply_chunk will return a dataframe as well.

    >>> def get_dataframe(x):
    ...     return pd.concat([x, x], axis=1)

    >>> s.mf.apply_chunk(get_dataframe, batch_rows=2).execute()
               0   1
    London    20  20
    New York  21  21
    Helsinki  12  12

    Provides a dtypes or dtype with name to naming the output schema.

    >>> s.mf.apply_chunk(
    ...    get_dataframe,
    ...    batch_rows=2,
    ...    dtypes={"A": np.int_, "B": np.int_},
    ...    output_type="dataframe"
    ... ).execute()
               A   B
    London    20  20
    New York  21  21
    Helsinki  12  12

    Create a series with a dict type.

    >>> import pyarrow as pa
    >>> from maxframe.lib.dtypes_extension import dict_
    >>> s = md.Series(
    ...     data=[[("k1", 1), ("k2", 2)], [("k1", 3)], None],
    ...     index=[1, 2, 3],
    ...     dtype=dict_(pa.string(), pa.int64()),
    ... )
    >>> s.execute()
    1    [('k1', 1), ('k2', 2)]
    2               [('k1', 3)]
    3                      <NA>
    dtype: map<string, int64>[pyarrow]

    Define a function that updates the map type with a new key-value pair in a batch.

    >>> def custom_set_item(row):
    ...     for _, value in row.items():
    ...         if value is not None:
    ...             value["x"] = 100
    ...     return row

    >>> s.mf.apply_chunk(
    ...     custom_set_item,
    ...     output_type="series",
    ...     dtype=s.dtype,
    ...     batch_rows=2,
    ...     skip_infer=True,
    ...     index=s.index,
    ... ).execute()
    1    [('k1', 1), ('k2', 2), ('x', 100)]
    2               [('k1', 3), ('x', 100)]
    3                                  <NA>
    dtype: map<string, int64>[pyarrow]
    """
    if not isinstance(func, Callable):
        raise TypeError("function must be a callable object")

    if not isinstance(batch_rows, int):
        raise TypeError("batch_rows must be an integer")

    if batch_rows <= 0:
        raise ValueError("batch_rows must be greater than 0")

    # bind args and kwargs
    output_types = kwargs.pop("output_types", None)
    object_type = kwargs.pop("object_type", None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type
    )
    output_type = output_types[0] if output_types else None
    if skip_infer and output_type is None:
        output_type = OutputType.df_or_series

    op = DataFrameApplyChunkOperator(
        func=func,
        batch_rows=batch_rows,
        output_type=output_type,
        args=args,
        kwargs=kwargs,
    )

    dtypes = (name, dtype) if dtype is not None else dtypes
    return op(
        dataframe_or_series,
        dtypes=dtypes,
        output_type=output_type,
        index=index,
    )
