# Copyright 1999-2026 Alibaba Group Holding Ltd.
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
from ...serialization.serializables import (
    DictField,
    FieldTypes,
    FunctionField,
    Int32Field,
    ListField,
    TupleField,
)
from ...udf import BuiltinFunction, MarkedFunction
from ...utils import (
    copy_if_possible,
    deprecate_positional_args,
    make_dtype,
    make_dtypes,
    pd_release_version,
)
from ..core import (
    DATAFRAME_GROUPBY_TYPE,
    GROUPBY_TYPE,
    DataFrameGroupBy,
    IndexValue,
    SeriesGroupBy,
)
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..type_infer import (
    InferredDataFrameMeta,
    infer_dataframe_return_value,
    prepend_group_keys_as_index,
)
from ..utils import (
    copy_func_scheduling_hints,
    make_column_list,
    parse_index,
    validate_output_types,
)
from .utils import warn_axis_argument, warn_prepend_index_group_keys

_apply_without_group_keys = pd_release_version < (1, 5, 0)
_has_include_groups = (2, 2, 0) <= pd_release_version < (3, 0, 0)


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
        skip_infer: bool = False,
        prepend_index_group_keys: bool = True,
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
            skip_infer=skip_infer,
            prepend_index_group_keys=prepend_index_group_keys,
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
        skip_infer: bool = False,
        prepend_index_group_keys: bool = True,
    ) -> InferredDataFrameMeta:
        def infer_func(groupby_obj):
            args = copy_if_possible(self.args or ())
            kwargs = copy_if_possible(self.kwargs or {})

            in_obj = input_groupby
            while isinstance(in_obj, GROUPBY_TYPE):
                in_obj = in_obj.inputs[0]

            by_cols = (
                make_column_list(self.groupby_params.get("by"), in_obj.dtypes) or []
            )
            if not self.groupby_params.get("selection"):
                selection = [
                    c for c in input_groupby.inputs[0].dtypes.index if c not in by_cols
                ]
                groupby_obj = groupby_obj[selection]
            if not _has_include_groups:
                kwargs.pop("include_groups", None)
            res = groupby_obj.apply(self.func, *args, **kwargs)
            if _apply_without_group_keys and not (
                prepend_index_group_keys
                or res.index.names != groupby_obj.obj.index.names
            ):
                # Need to patch group_index for legacy local pandas version
                #  only when index names not changed
                # FIXME here we add `not prepend_index_group_keys` to solely make
                #  our behavior consistent with legacy implementations. It should
                #  be removed once the argument is dropped
                res.index = prepend_group_keys_as_index(res.index, input_groupby)
            return res

        inferred_meta = infer_dataframe_return_value(
            input_groupby,
            infer_func,
            output_type=output_type,
            dtypes=dtypes,
            dtype=dtype,
            name=name,
            index=index,
            elementwise=elementwise,
            skip_infer=skip_infer,
            prepend_index_group_keys=prepend_index_group_keys,
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


@deprecate_positional_args
def df_groupby_apply_chunk(
    dataframe_groupby,
    func: Union[str, Callable],
    batch_rows=None,
    *,
    dtypes=None,
    dtype=None,
    name=None,
    output_type=None,
    index=None,
    skip_infer=False,
    order_cols=None,
    ascending=True,
    prepend_index_group_keys=False,
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
        returns a dataframe, a series or a scalar. In addition, the
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
        Whether to infer dtypes when dtypes or output_type is not specified.

    prepend_index_group_keys: bool, default False
        If True, the index of returned dataframe or series will automatically
        contain group keys if ``as_index=True``, or group indexes if
        ``as_index=False``, when ``group_keys=True``. It will also exclude
        group keys in user function inputs by default. See notes for more
        details.

        .. note::

            ``prepend_index_group_keys`` will be set to True by default in
            future releases, and a warning will be shown if the parameter
            is set to False. To make sure your code works in future
            releases, please set this to True and remove group indexes
            in index parameter or type annotation of ``func``.

    args, kwargs : tuple and dict
        Optional positional and keyword arguments to pass to ``func``.

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
      as ``dtypes`` of output DataFrame.
    * For Series output, you need to specify ``dtype`` and ``name`` of
      output Series.
    * ``index`` determines index of output DataFrame or Series. You may specify
      a dummy pandas index indicating the names and types of index of the output
      of ``func``, for instance, ``pd.MultiIndex.from_tuples([("a", 0)], names=["key1", "key2"])``.
      If ``index`` is not supplied, index of the input DataFrame or Series will
      be used. When `prepend_index_group_keys` is True, the index of the returning
      object will be ``index`` prepended with group information given ``as_index``
      and ``group_keys`` argument of the ``groupby`` function, which is consistent
      with pandas 3.0. When ``prepend_index_group_keys`` is False, you must specify
      a mock index with all fields, including group keys. As it is complicated to
      pass full index definition, ``prepend_index_group_keys=False`` will be
      deprecated in near future. Please supply ``prepend_index_group_keys=True``
      where possible.

    MaxFrame adopts expected behavior of pandas>=3.0 by ignoring group columns
    in user function input. If you still need a group column for your function
    input, try selecting it right after `groupby` results, for instance,
    ``df.groupby("A")[["A", "B", "C"]].mf.apply_chunk(func)`` will pass data of
    column A into ``func``.
    """
    if not prepend_index_group_keys:
        warn_prepend_index_group_keys(dataframe_groupby)
    else:
        kwargs = kwargs.copy()
        kwargs["include_groups"] = False
    warn_axis_argument("mf.apply_chunk", kwargs)

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
        output_type = (
            OutputType.dataframe if dataframe_groupby.ndim == 2 else OutputType.series
        )

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
        groupby_params=(dataframe_groupby.op.groupby_params or {}).copy(),
    )

    return op(
        dataframe_groupby,
        dtypes=dtypes,
        dtype=dtype,
        name=name,
        index=index,
        output_type=output_type,
        prepend_index_group_keys=prepend_index_group_keys,
    )
