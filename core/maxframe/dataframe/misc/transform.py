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

from typing import Any, MutableMapping, Union

import numpy as np
from pandas import DataFrame, Series

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import AnyField, BoolField, DictField, TupleField
from ...udf import BuiltinFunction, MarkedFunction
from ...utils import copy_if_possible, pd_release_version
from ..core import DATAFRAME_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import (
    InferredDataFrameMeta,
    build_df,
    build_series,
    copy_func_scheduling_hints,
    infer_dataframe_return_value,
    pack_func_args,
    parse_index,
    validate_axis,
)

_with_convert_dtype = pd_release_version < (1, 2, 0)


class DataFrameTransform(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.TRANSFORM
    _legacy_name = "TransformOperator"

    func = AnyField("func", default=None)
    axis = AnyField("axis", default=None)
    convert_dtype = BoolField("convert_dtype", default=None)
    args = TupleField("args", default=())
    kwds = DictField("kwds", default_factory=dict)

    call_agg = BoolField("call_agg", default=None)

    def __init__(self, output_types=None, memory_scale=None, **kw):
        super().__init__(_output_types=output_types, _memory_scale=memory_scale, **kw)
        if hasattr(self, "func"):
            copy_func_scheduling_hints(self.func, self)

    def has_custom_code(self) -> bool:
        return not isinstance(self.func, BuiltinFunction)

    def _infer_df_func_returns(
        self, df, dtypes=None, dtype=None, name=None, index=None
    ) -> InferredDataFrameMeta:
        def infer_func(df_obj):
            if self.call_agg:
                return df_obj.agg(self.func, self.axis)
            else:
                return df_obj.transform(self.func, self.axis)

        res = infer_dataframe_return_value(
            df,
            infer_func,
            self.output_types[0] if self.output_types else None,
            dtypes=dtypes,
            dtype=dtype,
            name=name,
            index=index,
            inherit_index=True,
        )
        res.check_absence("dtypes", "dtype")
        return res

    def __call__(
        self, df, dtypes=None, dtype=None, name=None, index=None, skip_infer=None
    ):
        axis = getattr(self, "axis", None) or 0
        self.axis = validate_axis(axis, df)
        if not skip_infer:
            inferred_meta = self._infer_df_func_returns(
                df, dtypes=dtypes, dtype=dtype, name=name, index=index
            )
        else:
            index_value = parse_index(index) if index else df.index_value
            inferred_meta = InferredDataFrameMeta(
                self.output_types[0],
                dtypes=dtypes,
                dtype=dtype,
                name=name,
                index_value=index_value,
            )

        self._output_types = [inferred_meta.output_type]
        if self.output_types[0] == OutputType.dataframe:
            new_shape = list(df.shape)
            new_index_value = inferred_meta.index_value
            dtypes = inferred_meta.dtypes
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
            if isinstance(df, DATAFRAME_TYPE):
                new_shape = (df.shape[1 - axis],)
                new_index_value = [df.columns_value, df.index_value][axis]
            else:
                new_shape = (np.nan,) if self.call_agg else df.shape
                new_index_value = df.index_value

            return self.new_series(
                [df],
                shape=new_shape,
                name=inferred_meta.name,
                dtype=inferred_meta.dtype,
                index_value=new_index_value,
            )

    @classmethod
    def estimate_size(
        cls, ctx: MutableMapping[str, Union[int, float]], op: "DataFrameTransform"
    ) -> None:
        if isinstance(op.func, MarkedFunction):
            ctx[op.outputs[0].key] = float("inf")
        super().estimate_size(ctx, op)


# keep for import compatibility
TransformOperator = DataFrameTransform


def get_packed_funcs(df, output_type, func, *args, **kwds) -> Any:
    stub_df = _build_stub_pandas_obj(df, output_type)
    n_args = copy_if_possible(args)
    n_kwds = copy_if_possible(kwds)
    return pack_func_args(stub_df, func, *n_args, **n_kwds)


def _build_stub_pandas_obj(df, output_type) -> Union[DataFrame, Series]:
    # TODO: Simulate a dataframe with the corresponding indexes if self.func is
    # a dict and axis=1
    if output_type == OutputType.dataframe:
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
    """
    call_agg = kwargs.pop("_call_agg", False)
    func = get_packed_funcs(df, OutputType.dataframe, func, *args, **kwargs)
    op = DataFrameTransform(
        func=func,
        axis=axis,
        args=args,
        kwds=kwargs,
        output_types=[OutputType.dataframe] if not call_agg else None,
        call_agg=call_agg,
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
    # FIXME: https://github.com/aliyun/alibabacloud-odps-maxframe-client/issues/10
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
    """
    call_agg = kwargs.pop("_call_agg", False)
    func = get_packed_funcs(series, OutputType.series, func, *args, **kwargs)
    op = DataFrameTransform(
        func=func,
        axis=axis,
        convert_dtype=convert_dtype,
        args=args,
        kwds=kwargs,
        output_types=[OutputType.series]
        if not call_agg and not isinstance(func, list)
        else None,
        call_agg=call_agg,
    )
    return op(series, dtype=dtype, name=series.name, skip_infer=skip_infer)
