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

from typing import Any, Callable, Dict, List, MutableMapping, Tuple, Union

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...lib.version import parse as parse_version
from ...serialization.serializables import (
    DictField,
    FieldTypes,
    FunctionField,
    Int32Field,
    ListField,
    TupleField,
)
from ...udf import BuiltinFunction, MarkedFunction
from ...utils import copy_if_possible, make_dtype, make_dtypes
from ..core import (
    DATAFRAME_GROUPBY_TYPE,
    GROUPBY_TYPE,
    INDEX_TYPE,
    DataFrameGroupBy,
    IndexValue,
    SeriesGroupBy,
)
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import (
    InferredDataFrameMeta,
    build_empty_df,
    copy_func_scheduling_hints,
    infer_dataframe_return_value,
    make_column_list,
    parse_index,
    validate_output_types,
)

_need_enforce_group_keys = parse_version(pd.__version__) < parse_version("1.5.0")


class GroupByApplyChunk(DataFrameOperatorMixin, DataFrameOperator):
    _op_type_ = opcodes.APPLY_CHUNK
    _op_module_ = "dataframe.groupby"

    func = FunctionField("func")
    batch_rows = Int32Field("batch_rows", default=None)
    args = TupleField("args", default=None)
    kwargs = DictField("kwargs", default=None)

    groupby_params = DictField("groupby_params", default=None)
    order_cols = ListField("order_cols", default=None)
    ascending = ListField("ascending", FieldTypes.bool, default_factory=lambda: [True])

    def __init__(self, output_type=None, **kw):
        if output_type:
            kw["_output_types"] = [output_type]
        super().__init__(**kw)
        if hasattr(self, "func"):
            copy_func_scheduling_hints(self.func, self)

    def has_custom_code(self) -> bool:
        return not isinstance(self.func, BuiltinFunction)

    def _call_dataframe(self, df, dtypes, dtype, name, index_value, element_wise):
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
        groupby: Union[DataFrameGroupBy, SeriesGroupBy],
        dtypes: Union[Tuple[str, Any], Dict[str, Any]] = None,
        dtype: Any = None,
        name: Any = None,
        output_type=None,
        index=None,
    ):
        input_df = groupby.inputs[0]
        if isinstance(input_df, GROUPBY_TYPE):
            input_df = input_df.inputs[0]

        # if skip_infer, directly build a frame
        if self.output_types and self.output_types[0] == OutputType.df_or_series:
            return self.new_df_or_series([input_df])

        # infer return index and dtypes
        inferred_meta = self._infer_batch_func_returns(
            groupby,
            output_type=output_type,
            dtypes=dtypes,
            dtype=dtype,
            name=name,
            index=index,
        )

        if inferred_meta.index_value is None:
            inferred_meta.index_value = parse_index(
                None, (groupby.key, groupby.index_value.key, self.func)
            )
        inferred_meta.check_absence("output_type", "dtypes", "dtype")

        if isinstance(groupby, DATAFRAME_GROUPBY_TYPE):
            return self._call_dataframe(
                input_df,
                dtypes=inferred_meta.dtypes,
                dtype=inferred_meta.dtype,
                name=inferred_meta.name,
                index_value=inferred_meta.index_value,
                element_wise=inferred_meta.elementwise,
            )

        return self._call_series(
            input_df,
            dtypes=inferred_meta.dtypes,
            dtype=inferred_meta.dtype,
            name=inferred_meta.name,
            index_value=inferred_meta.index_value,
            element_wise=inferred_meta.elementwise,
        )

    def _infer_batch_func_returns(
        self,
        input_groupby: Union[DataFrameGroupBy, SeriesGroupBy],
        output_type: OutputType,
        dtypes: Union[pd.Series, List[Any], Dict[str, Any]] = None,
        dtype: Any = None,
        name: Any = None,
        index: Union[pd.Index, IndexValue] = None,
        elementwise: bool = None,
    ) -> InferredDataFrameMeta:
        def infer_func(groupby_obj):
            args = copy_if_possible(self.args or ())
            kwargs = copy_if_possible(self.kwargs or {})

            in_obj = input_groupby
            while isinstance(in_obj, GROUPBY_TYPE):
                in_obj = in_obj.inputs[0]

            by_cols = make_column_list(groupby_params.get("by"), in_obj.dtypes) or []
            if not groupby_params.get("selection"):
                selection = [
                    c for c in input_groupby.inputs[0].dtypes.index if c not in by_cols
                ]
                groupby_obj = groupby_obj[selection]
            res = groupby_obj.apply(self.func, *args, **kwargs)
            if _need_enforce_group_keys and groupby_params.get("group_keys"):
                by_levels = (
                    make_column_list(groupby_params.get("level"), in_obj.index.names)
                    or []
                )

                input_df = input_groupby
                while isinstance(input_df, GROUPBY_TYPE):
                    input_df = input_df.inputs[0]

                idx_df = res.index.to_frame()
                if by_cols:
                    idx_names = by_cols + list(res.index.names)
                    mock_idx_df = build_empty_df(
                        input_df.dtypes[by_cols], index=idx_df.index
                    )
                else:
                    idx_names = by_levels + list(res.index.names)
                    if len(in_obj.index.names) > 1:
                        idx_dtypes = in_obj.index_value.value.dtypes
                    else:
                        idx_dtypes = pd.Series(
                            [in_obj.index.dtype], index=[in_obj.index.name]
                        )
                    mock_idx_df = build_empty_df(
                        idx_dtypes[by_levels], index=idx_df.index
                    )
                idx_df = pd.concat([mock_idx_df, idx_df], axis=1)
                res.index = pd.MultiIndex.from_frame(idx_df, names=idx_names)
            return res

        groupby_params = input_groupby.op.groupby_params
        inferred_meta = infer_dataframe_return_value(
            input_groupby,
            infer_func,
            output_type=output_type,
            dtypes=dtypes,
            dtype=dtype,
            name=name,
            index=index,
            elementwise=elementwise,
        )

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
                if index is not input_groupby.index_value
                else input_groupby.index_value
            )
        inferred_meta.elementwise = elementwise or inferred_meta.elementwise
        return inferred_meta

    @classmethod
    def estimate_size(
        cls,
        ctx: MutableMapping[str, Union[int, float]],
        op: "GroupByApplyChunk",
    ) -> None:
        if isinstance(op.func, MarkedFunction):
            ctx[op.outputs[0].key] = float("inf")
        super().estimate_size(ctx, op)


def df_groupby_apply_chunk(
    dataframe_groupby,
    func: Union[str, Callable],
    batch_rows=None,
    dtypes=None,
    dtype=None,
    name=None,
    output_type=None,
    index=None,
    skip_infer=False,
    order_cols=None,
    ascending=True,
    args=(),
    **kwargs,
):
    """
    Apply function `func` group-wise and combine the results together.
    The pandas DataFrame given to the function is a chunk of the input
    dataframe, consider as a batch rows.

    The function passed to `apply` must take a dataframe as its first
    argument and return a DataFrame, Series or scalar. `apply` will
    then take care of combining the results back together into a single
    dataframe or series. `apply` is therefore a highly flexible
    grouping method.

    Don't expect to receive all rows of the DataFrame in the function,
    as it depends on the implementation of MaxFrame and the internal
    running state of MaxCompute.

    Parameters
    ----------
    func : callable
        A callable that takes a dataframe as its first argument, and
        returns a dataframe, a series or a scalar. In addition the
        callable may take positional and keyword arguments.

    batch_rows : int
        Specify expected number of rows in a batch, as well as the len of
        function input dataframe. When the remaining data is insufficient,
        it may be less than this number.

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

    args, kwargs : tuple and dict
        Optional positional and keyword arguments to pass to `func`.

    Returns
    -------
    applied : Series or DataFrame

    See Also
    --------
    Series.apply : Apply a function to a Series.
    DataFrame.apply : Apply a function to each row or column of a DataFrame.
    DataFrame.mf.apply_chunk : Apply a function to row batches of a DataFrame.

    Notes
    -----
    When deciding output dtypes and shape of the return value, MaxFrame will
    try applying ``func`` onto a mock grouped object, and the apply call
    may fail. When this happens, you need to specify the type of apply
    call (DataFrame or Series) in output_type.

    * For DataFrame output, you need to specify a list or a pandas Series
      as ``dtypes`` of output DataFrame. ``index`` of output can also be
      specified.
    * For Series output, you need to specify ``dtype`` and ``name`` of
      output Series.

    MaxFrame adopts expected behavior of pandas>=3.0 by ignoring group columns
    in user function input. If you still need a group column for your function
    input, try selecting it right after `groupby` results, for instance,
    ``df.groupby("A")[["A", "B", "C"]].mf.apply_batch(func)`` will pass data of
    column A into ``func``.
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

    if order_cols and not isinstance(order_cols, list):
        order_cols = [order_cols]
    if not isinstance(ascending, list):
        ascending = [ascending]
    elif len(order_cols) != len(ascending):
        raise ValueError("order_cols and ascending must have same length")

    # bind args and kwargs
    op = GroupByApplyChunk(
        func=func,
        batch_rows=batch_rows,
        output_type=output_type,
        args=args,
        kwargs=kwargs,
        order_cols=order_cols,
        ascending=ascending,
        groupby_params=dataframe_groupby.op.groupby_params,
    )

    return op(
        dataframe_groupby,
        dtypes=dtypes,
        dtype=dtype,
        name=name,
        index=index,
        output_type=output_type,
    )
