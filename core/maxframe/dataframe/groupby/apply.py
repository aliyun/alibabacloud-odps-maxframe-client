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

from typing import MutableMapping, Union

import numpy as np

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
from ...udf import BuiltinFunction, MarkedFunction
from ...utils import copy_if_possible, get_func_token, make_dtype, make_dtypes, tokenize
from ..core import GROUPBY_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..type_infer import InferredDataFrameMeta, infer_dataframe_return_value
from ..utils import copy_func_scheduling_hints, parse_index, validate_output_types
from .utils import warn_axis_argument, warn_prepend_index_group_keys


class GroupByApplyLogicKeyGeneratorMixin(OperatorLogicKeyGeneratorMixin):
    def _get_logic_key_token_values(self):
        token_values = super()._get_logic_key_token_values()
        if self.func:
            return token_values + [get_func_token(self.func)]
        else:  # pragma: no cover
            return token_values


class GroupByApply(
    DataFrameOperator, DataFrameOperatorMixin, GroupByApplyLogicKeyGeneratorMixin
):
    _op_type_ = opcodes.APPLY
    _op_module_ = "dataframe.groupby"

    func = FunctionField("func")
    args = TupleField("args", default_factory=tuple)
    kwds = DictField("kwds", default_factory=dict)
    maybe_agg = BoolField("maybe_agg", default=None)

    logic_key = StringField("logic_key", default=None)
    func_ref = AnyField("func_ref", default=None)
    need_clean_up_func = BoolField("need_clean_up_func", default=False)
    groupby_params = DictField("groupby_params", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
        if hasattr(self, "func"):
            copy_func_scheduling_hints(self.func, self)

    def has_custom_code(self) -> bool:
        return not isinstance(self.func, BuiltinFunction)

    def _update_key(self):
        values = [v for v in self._values_ if v is not self.func] + [
            get_func_token(self.func)
        ]
        self._obj_set("_key", tokenize(type(self).__name__, *values))
        return self

    def _infer_df_func_returns(
        self,
        in_groupby,
        dtypes=None,
        dtype=None,
        name=None,
        index=None,
        prepend_index_group_keys=True,
    ) -> InferredDataFrameMeta:
        def infer_func(groupby_obj):
            args = copy_if_possible(self.args)
            kwds = copy_if_possible(self.kwds)
            return groupby_obj.apply(self.func, *args, **kwds)

        output_type = self.output_types[0] if self.output_types else None
        inferred_meta = infer_dataframe_return_value(
            in_groupby,
            infer_func,
            dtypes=dtypes,
            dtype=dtype,
            name=name,
            index=index,
            output_type=output_type,
            prepend_index_group_keys=prepend_index_group_keys,
        )
        self.output_types = (
            [inferred_meta.output_type]
            if not self.output_types and inferred_meta.output_type
            else self.output_types
        )
        self.maybe_agg = inferred_meta.maybe_agg
        return inferred_meta

    def __call__(
        self,
        groupby,
        dtypes=None,
        dtype=None,
        name=None,
        index=None,
        prepend_index_group_keys=True,
    ):
        input_df = groupby.inputs[0]
        if isinstance(input_df, GROUPBY_TYPE):
            input_df = input_df.inputs[0]

        if self.output_types and self.output_types[0] == OutputType.df_or_series:
            return self.new_df_or_series([input_df])

        inferred_meta = self._infer_df_func_returns(
            groupby,
            dtypes=dtypes,
            dtype=dtype,
            name=name,
            index=index,
            prepend_index_group_keys=prepend_index_group_keys,
        )
        inferred_meta.check_absence("output_type", "dtypes", "dtype")
        if self.output_types[0] == OutputType.dataframe:
            new_shape = (np.nan, len(inferred_meta.dtypes))
            return self.new_dataframe(
                [input_df],
                shape=new_shape,
                dtypes=inferred_meta.dtypes,
                index_value=inferred_meta.index_value,
                columns_value=parse_index(inferred_meta.dtypes.index, store_data=True),
            )
        else:
            new_shape = (np.nan,)
            return self.new_series(
                [input_df],
                name=inferred_meta.name,
                shape=new_shape,
                dtype=inferred_meta.dtype,
                index_value=inferred_meta.index_value,
            )

    @classmethod
    def estimate_size(
        cls, ctx: MutableMapping[str, Union[int, float]], op: "GroupByApply"
    ) -> None:
        if isinstance(op.func, MarkedFunction):
            ctx[op.outputs[0].key] = float("inf")
        super().estimate_size(ctx, op)


def groupby_apply(
    in_groupby,
    func,
    *args,
    output_type=None,
    dtypes=None,
    dtype=None,
    name=None,
    index=None,
    skip_infer=None,
    prepend_index_group_keys=False,
    **kwargs,
):
    """
    Apply function `func` group-wise and combine the results together.

    The function passed to `apply` must take a dataframe as its first
    argument and return a DataFrame, Series or scalar. `apply` will
    then take care of combining the results back together into a single
    dataframe or series. `apply` is therefore a highly flexible
    grouping method.

    While `apply` is a very flexible method, its downside is that
    using it can be quite a bit slower than using more specific methods
    like `agg` or `transform`. Pandas offers a wide range of method that will
    be much faster than using `apply` for their specific purposes, so try to
    use them before reaching for `apply`.

    Parameters
    ----------
    func : callable
        A callable that takes a dataframe as its first argument, and
        returns a dataframe, a series or a scalar. In addition, the
        callable may take positional and keyword arguments.

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
        contain group keys if as_index=True, or group indexes if as_index=False,
        when group_keys=True.

        .. note::
            `prepend_index_group_keys` will be set to True by default in
            future releases, and a warning will be shown if the parameter
            is set to False. To make sure your code works in future
            releases, please set this to True and remove group indexes
            in index parameter or type annotation of `func`.

    args, kwargs : tuple and dict
        Optional positional and keyword arguments to pass to `func`.

    Returns
    -------
    applied : Series or DataFrame

    See Also
    --------
    pipe : Apply function to the full GroupBy object instead of to each
        group.
    aggregate : Apply aggregate function to the GroupBy object.
    transform : Apply function column-by-column to the GroupBy object.
    Series.apply : Apply a function to a Series.
    DataFrame.apply : Apply a function to each row or column of a DataFrame.

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
    """
    if not prepend_index_group_keys:
        warn_prepend_index_group_keys(in_groupby)
    warn_axis_argument("apply", kwargs)

    output_types = kwargs.pop("output_types", None)
    object_type = kwargs.pop("object_type", None)
    output_types = validate_output_types(
        output_types=output_types, output_type=output_type, object_type=object_type
    )
    if skip_infer and output_types is None:
        output_types = [
            OutputType.dataframe if in_groupby.ndim == 2 else OutputType.series
        ]

    dtypes = make_dtypes(dtypes)
    dtype = make_dtype(dtype)
    op = GroupByApply(
        func=func,
        args=args,
        kwds=kwargs,
        output_types=output_types,
        groupby_params=(in_groupby.op.groupby_params or {}).copy(),
    )
    return op(
        in_groupby,
        dtypes=dtypes,
        dtype=dtype,
        name=name,
        index=index,
        prepend_index_group_keys=prepend_index_group_keys,
    )
