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

import numpy as np
import pandas as pd

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
    copy_func_scheduling_hints,
    make_dtype,
    make_dtypes,
    parse_index,
    validate_output_types,
)


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
    func_key = AnyField("func_key", default=None)
    need_clean_up_func = BoolField("need_clean_up_func", default=False)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
        if hasattr(self, "func"):
            copy_func_scheduling_hints(self.func, self)

    def _update_key(self):
        values = [v for v in self._values_ if v is not self.func] + [
            get_func_token(self.func)
        ]
        self._obj_set("_key", tokenize(type(self).__name__, *values))
        return self

    def _infer_df_func_returns(
        self, in_groupby, in_df, dtypes=None, dtype=None, name=None, index=None
    ):
        index_value, output_type, new_dtypes = None, None, None

        if self.output_types is not None and (dtypes is not None or dtype is not None):
            ret_dtypes = dtypes if dtypes is not None else (dtype, name)
            ret_index_value = parse_index(index) if index is not None else None
            return ret_dtypes, ret_index_value

        try:
            infer_df = in_groupby.op.build_mock_groupby().apply(
                self.func, *self.args, **self.kwds
            )

            if len(infer_df) <= 2:
                # we create mock df with 4 rows, 2 groups
                # if return df has 2 rows, we assume that
                # it's an aggregation operation
                self.maybe_agg = True

            # todo return proper index when sort=True is implemented
            index_value = parse_index(infer_df.index[:0], in_df.key, self.func)

            # for backward compatibility
            dtype = dtype if dtype is not None else dtypes
            if isinstance(infer_df, pd.DataFrame):
                output_type = output_type or OutputType.dataframe
                new_dtypes = new_dtypes or infer_df.dtypes
            elif isinstance(infer_df, pd.Series):
                output_type = output_type or OutputType.series
                new_dtypes = new_dtypes or (
                    name or infer_df.name,
                    dtype or infer_df.dtype,
                )
            else:
                output_type = OutputType.series
                new_dtypes = (name, dtype or pd.Series(infer_df).dtype)
        except:  # noqa: E722  # nosec
            pass

        self.output_types = (
            [output_type]
            if not self.output_types and output_type
            else self.output_types
        )
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        return dtypes, index_value

    def __call__(self, groupby, dtypes=None, dtype=None, name=None, index=None):
        in_df = groupby
        if self.output_types and self.output_types[0] == OutputType.df_or_series:
            return self.new_df_or_series([groupby])
        while in_df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
            in_df = in_df.inputs[0]

        with quiet_stdio():
            dtypes, index_value = self._infer_df_func_returns(
                groupby, in_df, dtypes, dtype=dtype, name=name, index=index
            )
        if index_value is None:
            index_value = parse_index(None, (in_df.key, in_df.index_value.key))
        for arg, desc in zip((self.output_types, dtypes), ("output_types", "dtypes")):
            if arg is None:
                raise TypeError(
                    f"Cannot determine {desc} by calculating with enumerate data, "
                    "please specify it as arguments"
                )

        if self.output_types[0] == OutputType.dataframe:
            new_shape = (np.nan, len(dtypes))
            return self.new_dataframe(
                [groupby],
                shape=new_shape,
                dtypes=dtypes,
                index_value=index_value,
                columns_value=parse_index(dtypes.index, store_data=True),
            )
        else:
            name = name or dtypes[0]
            dtype = dtype or dtypes[1]
            new_shape = (np.nan,)
            return self.new_series(
                [groupby],
                name=name,
                shape=new_shape,
                dtype=dtype,
                index_value=index_value,
            )


def groupby_apply(
    groupby,
    func,
    *args,
    output_type=None,
    dtypes=None,
    dtype=None,
    name=None,
    index=None,
    skip_infer=None,
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
        returns a dataframe, a series or a scalar. In addition the
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
        Whether infer dtypes when dtypes or output_type is not specified.

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
    output_types = kwargs.pop("output_types", None)
    object_type = kwargs.pop("object_type", None)
    output_types = validate_output_types(
        output_types=output_types, output_type=output_type, object_type=object_type
    )
    if output_types is None and skip_infer:
        output_types = [OutputType.df_or_series]

    dtypes = make_dtypes(dtypes)
    dtype = make_dtype(dtype)
    op = GroupByApply(func=func, args=args, kwds=kwargs, output_types=output_types)
    return op(groupby, dtypes=dtypes, dtype=dtype, name=name, index=index)
