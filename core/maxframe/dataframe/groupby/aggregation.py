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
import logging
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ... import opcodes
from ...config import options
from ...core import ENTITY_TYPE, EntityData, OutputType, enter_mode
from ...serialization import PickleContainer
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    Int8Field,
    Int32Field,
    Int64Field,
    ListField,
    StringField,
)
from ...udf import BuiltinFunction
from ...utils import find_objects, get_pd_option, lazy_import, pd_release_version
from ..core import GROUPBY_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..reduction.aggregation import (
    compile_reduction_funcs,
    is_funcs_aggregate,
    normalize_reduction_funcs,
)
from ..utils import is_cudf, parse_index

cp = lazy_import("cupy", rename="cp")
cudf = lazy_import("cudf")

logger = logging.getLogger(__name__)
CV_THRESHOLD = 0.2
MEAN_RATIO_THRESHOLD = 2 / 3
_support_get_group_without_as_index = pd_release_version[:2] > (1, 0)
_support_multi_index_as_index = pd_release_version[:2] > (2, 0)


_agg_functions = {
    "sum": lambda x: x.sum(),
    "prod": lambda x: x.prod(),
    "product": lambda x: x.product(),
    "min": lambda x: x.min(),
    "max": lambda x: x.max(),
    "all": lambda x: x.all(),
    "any": lambda x: x.any(),
    "count": lambda x: x.count(),
    "size": lambda x: x._reduction_size(),
    "mean": lambda x: x.mean(),
    "var": lambda x, ddof=1: x.var(ddof=ddof),
    "std": lambda x, ddof=1: x.std(ddof=ddof),
    "sem": lambda x, ddof=1: x.sem(ddof=ddof),
    "skew": lambda x, bias=False: x.skew(bias=bias),
    "kurt": lambda x, bias=False: x.kurt(bias=bias),
    "kurtosis": lambda x, bias=False: x.kurtosis(bias=bias),
    "nunique": lambda x: x.nunique(),
    "median": lambda x: x.median(),
}
_series_col_name = "col_name"


def _patch_groupby_kurt():
    try:
        try:
            from pandas.api.typing import DataFrameGroupBy, SeriesGroupBy
        except ImportError:
            from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

        if hasattr(DataFrameGroupBy, "kurt"):  # pragma: no branch
            return

        def _kurt_by_frame(a, *args, **kwargs):
            data = a.to_frame().kurt(*args, **kwargs).iloc[0]
            if is_cudf(data):  # pragma: no cover
                data = data.copy()
            return data

        def _group_kurt(x, *args, **kwargs):
            if kwargs.get("numeric_only") is not None:
                return x.agg(functools.partial(_kurt_by_frame, *args, **kwargs))
            else:
                return x.agg(functools.partial(pd.Series.kurt, *args, **kwargs))

        DataFrameGroupBy.kurt = DataFrameGroupBy.kurtosis = _group_kurt
        SeriesGroupBy.kurt = SeriesGroupBy.kurtosis = _group_kurt
    except (AttributeError, ImportError):  # pragma: no cover
        pass


_patch_groupby_kurt()
del _patch_groupby_kurt


def build_mock_agg_result(
    groupby: GROUPBY_TYPE,
    groupby_params: Dict,
    raw_func: Callable,
    **raw_func_kw,
):
    try:
        with enter_mode(mock=True):
            agg_result = groupby.op.build_mock_groupby().aggregate(
                raw_func, **raw_func_kw
            )
    except ValueError:
        if (
            groupby_params.get("as_index") or _support_get_group_without_as_index
        ):  # pragma: no cover
            raise
        agg_result = (
            groupby.op.build_mock_groupby(as_index=True)
            .aggregate(raw_func, **raw_func_kw)
            .to_frame()
        )
        agg_result.index.names = [None] * agg_result.index.nlevels
    return agg_result


class DataFrameGroupByAgg(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.GROUPBY_AGG

    raw_func = AnyField("raw_func", default=None)
    raw_func_kw = DictField("raw_func_kw", default=None)
    func = AnyField("func", default=None)
    func_rename = ListField("func_rename", default=None)

    raw_groupby_params = DictField("raw_groupby_params", default=None)
    groupby_params = DictField("groupby_params", default=None)

    method = StringField("method", default=None)

    # for chunk
    chunk_store_limit = Int64Field("chunk_store_limit", default=None)
    pre_funcs = ListField("pre_funcs", default=None)
    agg_funcs = ListField("agg_funcs", default=None)
    post_funcs = ListField("post_funcs", default=None)
    index_levels = Int32Field("index_levels", default=None)
    size_recorder_name = StringField("size_recorder_name", default=None)
    combine_size = Int32Field("combine_size", default=None)

    use_inf_as_na = BoolField("use_inf_as_na", default=None)
    input_ndim = Int8Field("input_ndim", default=1)
    append_level = BoolField("append_level", default=False)

    def has_custom_code(self) -> bool:
        callable_bys = find_objects(
            self.groupby_params.get("by"), types=PickleContainer, checker=callable
        )
        if callable_bys and any(
            not isinstance(fun, BuiltinFunction) for fun in callable_bys
        ):
            return True

        return any(
            fun.custom_reduction
            and not isinstance(fun.custom_reduction, BuiltinFunction)
            for fun in self.agg_funcs or ()
        )

    @classmethod
    def _set_inputs(cls, op: "DataFrameGroupByAgg", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs[1:])
        if len(op._inputs) > 1:
            by = []
            for v in op.groupby_params["by"]:
                if isinstance(v, ENTITY_TYPE):
                    by.append(next(inputs_iter))
                else:
                    by.append(v)
            op.groupby_params["by"] = by

    def _get_inputs(self, inputs):
        if isinstance(self.groupby_params["by"], list):
            for v in self.groupby_params["by"]:
                if isinstance(v, ENTITY_TYPE):
                    inputs.append(v)
        return inputs

    def _get_index_levels(self, groupby, mock_index):
        if not self.groupby_params["as_index"]:
            try:
                as_index_agg_df = groupby.op.build_mock_groupby(
                    as_index=True
                ).aggregate(self.raw_func, **self.raw_func_kw)
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                # handling cases like mdf.groupby("b", as_index=False).b.agg({"c": "count"})
                if isinstance(self.groupby_params["by"], list):
                    return len(self.groupby_params["by"])
                raise  # pragma: no cover
            pd_index = as_index_agg_df.index
        else:
            pd_index = mock_index
        return 1 if not isinstance(pd_index, pd.MultiIndex) else len(pd_index.levels)

    def _fix_as_index(self, result_index: pd.Index):
        # make sure if as_index=False takes effect
        if not _support_multi_index_as_index and isinstance(
            result_index, pd.MultiIndex
        ):
            # if MultiIndex, as_index=False definitely takes no effect
            self.groupby_params["as_index"] = True
        elif result_index.name is not None:
            # if not MultiIndex and agg_df.index has a name
            # means as_index=False takes no effect
            self.groupby_params["as_index"] = True

    def _call_dataframe(self, groupby, input_df):
        compile_reduction_funcs(self, input_df)
        agg_df = build_mock_agg_result(
            groupby, self.groupby_params, self.raw_func, **self.raw_func_kw
        )

        shape = (np.nan, agg_df.shape[1])
        if isinstance(agg_df.index, pd.RangeIndex):
            index_value = parse_index(
                pd.RangeIndex(-1), groupby.key, groupby.index_value.key
            )
        else:
            index_value = parse_index(
                agg_df.index, groupby.key, groupby.index_value.key
            )

        self.input_ndim = 2

        # make sure if as_index=False takes effect
        self._fix_as_index(agg_df.index)

        # determine num of indices to group in intermediate steps
        self.index_levels = self._get_index_levels(groupby, agg_df.index)

        # if True, name of agg funcs will be appended as the last level
        self.append_level = agg_df.dtypes.index.nlevels > input_df.dtypes.index.nlevels

        inputs = self._get_inputs([input_df])
        return self.new_dataframe(
            inputs,
            shape=shape,
            dtypes=agg_df.dtypes,
            index_value=index_value,
            columns_value=parse_index(agg_df.columns, store_data=True),
        )

    def _call_series(self, groupby, in_series):
        compile_reduction_funcs(self, in_series)
        agg_result = build_mock_agg_result(
            groupby, self.groupby_params, self.raw_func, **self.raw_func_kw
        )

        # make sure if as_index=False takes effect
        self._fix_as_index(agg_result.index)

        index_value = parse_index(
            agg_result.index, groupby.key, groupby.index_value.key
        )

        inputs = self._get_inputs([in_series])

        self.input_ndim = 1

        # determine num of indices to group in intermediate steps
        self.index_levels = self._get_index_levels(groupby, agg_result.index)

        # update value type
        if isinstance(agg_result, pd.DataFrame):
            return self.new_dataframe(
                inputs,
                shape=(np.nan, len(agg_result.columns)),
                dtypes=agg_result.dtypes,
                index_value=index_value,
                columns_value=parse_index(agg_result.columns, store_data=True),
            )
        else:
            return self.new_series(
                inputs,
                shape=(np.nan,),
                dtype=agg_result.dtype,
                name=agg_result.name,
                index_value=index_value,
            )

    def __call__(self, groupby):
        normalize_reduction_funcs(self, ndim=groupby.ndim)
        df = groupby
        while df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
            df = df.inputs[0]

        if self.raw_func == "size":
            self.output_types = [OutputType.series]
        else:
            self.output_types = (
                [OutputType.dataframe]
                if groupby.op.output_types[0] == OutputType.dataframe_groupby
                else [OutputType.series]
            )

        if self.output_types[0] == OutputType.dataframe:
            return self._call_dataframe(groupby, df)
        else:
            return self._call_series(groupby, df)


def agg(groupby, func=None, method="auto", *args, **kwargs):
    """
    Aggregate using one or more operations on grouped data.

    Parameters
    ----------
    groupby : MaxFrame Groupby
        Groupby data.
    func : str or list-like
        Aggregation functions.
    method : {'auto', 'shuffle', 'tree'}, default 'auto'
        'tree' method provide a better performance, 'shuffle' is recommended
        if aggregated result is very large, 'auto' will use 'shuffle' method
        in distributed mode and use 'tree' in local mode.

    Returns
    -------
    Series or DataFrame
        Aggregated result.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame(
    ...     {
    ...         "A": [1, 1, 2, 2],
    ...         "B": [1, 2, 3, 4],
    ...         "C": [0.362838, 0.227877, 1.267767, -0.562860],
    ...     }
    ... ).execute()
       A  B         C
    0  1  1  0.362838
    1  1  2  0.227877
    2  2  3  1.267767
    3  2  4 -0.562860

    The aggregation is for each column.

    >>> df.groupby('A').agg('min').execute()
       B         C
    A
    1  1  0.227877
    2  3 -0.562860

    Multiple aggregations.

    >>> df.groupby('A').agg(['min', 'max']).execute()
        B             C
      min max       min       max
    A
    1   1   2  0.227877  0.362838
    2   3   4 -0.562860  1.267767

    Different aggregations per column

    >>> df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'}).execute()
        B             C
      min max       sum
    A
    1   1   2  0.590715
    2   3   4  0.704907

    To control the output names with different aggregations per column,
    MaxFrame supports “named aggregation”

    >>> from maxframe.dataframe import NamedAgg
    >>> df.groupby("A").agg(
    ...  b_min=NamedAgg(column="B", aggfunc="min"),
    ...  c_sum=NamedAgg(column="C", aggfunc="sum")).execute()
       b_min     c_sum
    A
    1      1  0.590715
    2      3  0.704907
    """

    # When perform a computation on the grouped data, we won't shuffle
    # the data in the stage of groupby and do shuffle after aggregation.

    if not isinstance(groupby, GROUPBY_TYPE):
        raise TypeError(f"Input should be type of groupby, not {type(groupby)}")

    if method is None:
        method = "auto"
    if method not in ["shuffle", "tree", "auto"]:
        raise ValueError(
            f"Method {method} is not available, please specify 'tree' or 'shuffle"
        )

    combine_size = (
        kwargs.pop("combine_size", None) or options.dpe.reduction.combine_size
    )

    if not is_funcs_aggregate(func, ndim=groupby.ndim):
        # pass index to transform, otherwise it will lose name info for index
        agg_result = build_mock_agg_result(
            groupby, groupby.op.groupby_params, func, **kwargs
        )
        if isinstance(agg_result.index, pd.RangeIndex):
            # set -1 to represent unknown size for RangeIndex
            index_value = parse_index(
                pd.RangeIndex(-1), groupby.key, groupby.index_value.key
            )
        else:
            index_value = parse_index(
                agg_result.index, groupby.key, groupby.index_value.key
            )
        return groupby.transform(
            func, *args, _call_agg=True, index=index_value, **kwargs
        )

    agg_op = DataFrameGroupByAgg(
        raw_func=func,
        raw_func_kw=kwargs,
        method=method,
        raw_groupby_params=groupby.op.groupby_params,
        groupby_params=groupby.op.groupby_params,
        combine_size=combine_size,
        chunk_store_limit=options.chunk_store_limit,
        use_inf_as_na=get_pd_option("mode.use_inf_as_na", False),
    )
    return agg_op(groupby)
