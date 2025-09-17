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
from typing import Any, Callable, Dict, List, MutableMapping, Tuple, Union

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
from ...typing_ import TileableType
from ...udf import BuiltinFunction, MarkedFunction
from ...utils import copy_if_possible, make_dtype, make_dtypes
from ..core import DATAFRAME_TYPE, INDEX_TYPE, DataFrame, IndexValue, Series
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import (
    InferredDataFrameMeta,
    build_df,
    copy_func_scheduling_hints,
    infer_dataframe_return_value,
    pack_func_args,
    parse_index,
    validate_output_types,
)


class DataFrameApplyChunk(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.APPLY_CHUNK
    _legacy_name = "DataFrameApplyChunkOperator"  # since v2.0.0

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

    def has_custom_code(self) -> bool:
        return not isinstance(self.func, BuiltinFunction)

    def check_inputs(self, inputs: List[TileableType]):
        # for apply_chunk we allow called on non-deterministic tileables
        pass

    def _call_dataframe(self, df, dtypes, dtype, name, index_value, element_wise):
        # return dataframe
        if self.output_types[0] == OutputType.dataframe:
            dtypes = make_dtypes(dtypes)
            if dtypes is not None:
                shape = df.shape if element_wise else (np.nan, len(dtypes))
                cols_value = parse_index(dtypes.index, store_data=True)
            else:
                shape = (np.nan, np.nan)
                cols_value = None
            # apply_chunk will use generate new range index for results
            return self.new_dataframe(
                [df],
                shape=shape,
                index_value=index_value,
                columns_value=cols_value,
                dtypes=dtypes,
            )

        # return series
        return self.new_series(
            [df], shape=(np.nan,), name=name, dtype=dtype, index_value=index_value
        )

    def _call_series(self, series, dtypes, dtype, name, index_value, element_wise):
        if self.output_types[0] == OutputType.series:
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
        dtype: Any = None,
        name: Any = None,
        output_type=None,
        index=None,
        skip_infer=False,
    ):
        args = self.args or ()
        kwargs = self.kwargs or {}
        # if not dtypes and not skip_infer:
        try:
            packed_func = get_packed_func(df_or_series, self.func, *args, **kwargs)
        except:
            if not skip_infer:
                raise
            packed_func = self.func

        # if skip_infer, directly build a frame
        if self.output_types and self.output_types[0] == OutputType.df_or_series:
            return self.new_df_or_series([df_or_series])

        # infer return index and dtypes
        inferred_meta = self._infer_batch_func_returns(
            df_or_series,
            packed_func=packed_func,
            output_type=output_type,
            dtypes=dtypes,
            dtype=dtype,
            name=name,
            index=index,
            skip_infer=skip_infer,
        )

        if inferred_meta.index_value is None:
            inferred_meta.index_value = parse_index(
                None, (df_or_series.key, df_or_series.index_value.key, self.func)
            )
        if not skip_infer:
            inferred_meta.check_absence("output_type", "dtypes", "dtype")

        if isinstance(df_or_series, DATAFRAME_TYPE):
            return self._call_dataframe(
                df_or_series,
                dtypes=inferred_meta.dtypes,
                dtype=inferred_meta.dtype,
                name=inferred_meta.name,
                index_value=inferred_meta.index_value,
                element_wise=inferred_meta.elementwise,
            )

        return self._call_series(
            df_or_series,
            dtypes=inferred_meta.dtypes,
            dtype=inferred_meta.dtype,
            name=inferred_meta.name,
            index_value=inferred_meta.index_value,
            element_wise=inferred_meta.elementwise,
        )

    def _infer_batch_func_returns(
        self,
        input_df_or_series: Union[DataFrame, Series],
        packed_func: Union[Callable, functools.partial],
        output_type: OutputType,
        *args,
        dtypes: Union[pd.Series, List[Any], Dict[str, Any]] = None,
        dtype: Any = None,
        name: Any = None,
        index: Union[pd.Index, IndexValue] = None,
        elementwise: bool = None,
        skip_infer: bool = False,
        **kwargs,
    ) -> InferredDataFrameMeta:
        inferred_meta = infer_dataframe_return_value(
            input_df_or_series,
            functools.partial(packed_func, *args, **kwargs),
            output_type=output_type,
            dtypes=dtypes,
            dtype=dtype,
            name=name,
            index=index,
            elementwise=elementwise,
            skip_infer=skip_infer,
        )
        if skip_infer:
            return inferred_meta

        # merge specified and inferred index, dtypes, output_type
        # elementwise used to decide shape
        self.output_types = (
            [inferred_meta.output_type]
            if not self.output_types and inferred_meta.output_type
            else self.output_types
        )
        if self.output_types:
            inferred_meta.output_type = self.output_types[0]
        inferred_meta.dtypes = dtypes if dtypes is not None else inferred_meta.dtypes
        if isinstance(index, INDEX_TYPE):
            index = index.index_value
        if index is not None:
            inferred_meta.index_value = (
                parse_index(index)
                if index is not input_df_or_series.index_value
                else input_df_or_series.index_value
            )
        inferred_meta.elementwise = elementwise or inferred_meta.elementwise
        return inferred_meta

    @classmethod
    def estimate_size(
        cls,
        ctx: MutableMapping[str, Union[int, float]],
        op: "DataFrameApplyChunk",
    ) -> None:
        if isinstance(op.func, MarkedFunction):
            ctx[op.outputs[0].key] = float("inf")
        super().estimate_size(ctx, op)


# Keep for import compatibility
DataFrameApplyChunkOperator = DataFrameApplyChunk


def get_packed_func(df, func, *args, **kwargs) -> Any:
    stub_df = build_df(df, fill_value=1, size=1)
    n_args = copy_if_possible(args)
    n_kwargs = copy_if_possible(kwargs)
    return pack_func_args(stub_df, func, *n_args, **n_kwargs)


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

    if dtype is not None:
        dtype = make_dtype(dtype)

    output_types = kwargs.pop("output_types", None)
    object_type = kwargs.pop("object_type", None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type
    )
    output_type = output_types[0] if output_types else None
    if skip_infer and output_type is None:
        output_type = OutputType.df_or_series

    # bind args and kwargs
    op = DataFrameApplyChunk(
        func=func,
        batch_rows=batch_rows,
        output_type=output_type,
        args=args,
        kwargs=kwargs,
    )

    return op(
        dataframe,
        dtypes=dtypes,
        dtype=dtype,
        name=name,
        index=index,
        output_type=output_type,
        skip_infer=skip_infer,
    )


def series_apply_chunk(
    dataframe_or_series,
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

    if batch_rows is not None:
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

    op = DataFrameApplyChunk(
        func=func,
        batch_rows=batch_rows,
        output_type=output_type,
        args=args,
        kwargs=kwargs,
    )

    if dtype is not None:
        dtype = make_dtype(dtype)
    return op(
        dataframe_or_series,
        dtypes=make_dtypes(dtypes),
        dtype=dtype,
        name=name,
        output_type=output_type,
        index=index,
    )
