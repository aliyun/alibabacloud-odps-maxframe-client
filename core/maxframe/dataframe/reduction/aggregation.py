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

import copy
import functools
import inspect
import itertools
from collections import OrderedDict
from collections.abc import Iterable
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa

from ... import opcodes
from ... import tensor as maxframe_tensor
from ...core import ENTITY_TYPE, OutputType, enter_mode
from ...io.odpsio.schema import pandas_dtype_to_arrow_type
from ...lib.dtypes_extension import ArrowDtype
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    Int32Field,
    ListField,
)
from ...typing_ import TileableType
from ...udf import BuiltinFunction
from ...utils import get_pd_option, lazy_import, pd_release_version
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_df, build_empty_df, build_series, parse_index, validate_axis
from .core import (
    CustomReduction,
    ReductionAggStep,
    ReductionCompiler,
    ReductionPostStep,
    ReductionPreStep,
)
from .unique import _unique

cp = lazy_import("cupy", rename="cp")
cudf = lazy_import("cudf")

_agg_size_as_series = pd_release_version >= (1, 3, 0)


def where_function(cond, var1, var2):
    if hasattr(var1, "ndim") and var1.ndim >= 1:
        return var1.where(cond, var2)
    elif isinstance(var1, ENTITY_TYPE):
        return maxframe_tensor.where(cond, var1, var2)
    else:
        return np.where(cond, var1, var2).item()


_agg_functions = {
    "sum": lambda x, skipna=True: x.sum(skipna=skipna),
    "prod": lambda x, skipna=True: x.prod(skipna=skipna),
    "product": lambda x, skipna=True: x.product(skipna=skipna),
    "min": lambda x, skipna=True: x.min(skipna=skipna),
    "max": lambda x, skipna=True: x.max(skipna=skipna),
    "all": lambda x, skipna=True: x.all(skipna=skipna),
    "any": lambda x, skipna=True: x.any(skipna=skipna),
    "count": lambda x: x.count(),
    "size": lambda x: x._reduction_size(),
    "mean": lambda x, skipna=True: x.mean(skipna=skipna),
    "var": lambda x, skipna=True, ddof=1: x.var(skipna=skipna, ddof=ddof),
    "std": lambda x, skipna=True, ddof=1: x.std(skipna=skipna, ddof=ddof),
    "sem": lambda x, skipna=True, ddof=1: x.sem(skipna=skipna, ddof=ddof),
    "skew": lambda x, skipna=True, bias=False: x.skew(skipna=skipna, bias=bias),
    "kurt": lambda x, skipna=True, bias=False: x.kurt(skipna=skipna, bias=bias),
    "kurtosis": lambda x, skipna=True, bias=False: x.kurtosis(skipna=skipna, bias=bias),
    "nunique": lambda x: x.nunique(),
    "unique": lambda x: _unique(x, output_list_scalar=True),
    "median": lambda x, skipna=True: x.median(skipna=skipna),
}


class DataFrameAggregate(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.AGGREGATE

    raw_func = AnyField("raw_func")
    raw_func_kw = DictField("raw_func_kw", default=None)
    func = AnyField("func", default=None)
    func_rename = ListField("func_rename", default=None)
    axis = AnyField("axis", default=0)
    numeric_only = BoolField("numeric_only", default=None)
    bool_only = BoolField("bool_only", default=None)

    pre_funcs: List[ReductionPreStep] = ListField("pre_funcs", default=None)
    agg_funcs: List[ReductionAggStep] = ListField("agg_funcs", default=None)
    post_funcs: List[ReductionPostStep] = ListField("post_funcs", default=None)
    combine_size = Int32Field("combine_size", default=None)
    use_inf_as_na = BoolField("use_inf_as_na", default=False)

    def has_custom_code(self) -> bool:
        return any(
            fun.custom_reduction
            and not isinstance(fun.custom_reduction, BuiltinFunction)
            for fun in self.agg_funcs or ()
        )

    @staticmethod
    def _filter_dtypes(op: "DataFrameAggregate", dtypes):
        if not op.numeric_only and not op.bool_only:
            return dtypes
        empty_df = build_empty_df(dtypes)
        return empty_df.select_dtypes(
            [np.number, np.bool_] if op.numeric_only else [np.bool_]
        ).dtypes

    def _fill_df_dtypes(self, in_df, dtypes):
        if all(dt != np.dtype("O") for dt in dtypes):
            return dtypes

        if isinstance(self.func, dict):
            col_func_it = self.func.items()
        else:
            assert in_df.ndim == 2
            col_func_it = itertools.product(in_df.dtypes.index, self.func)

        col_to_dt = dict(in_df.dtypes.items())

        new_dt = OrderedDict()
        for (col_name, func), (out_col_name, dt) in zip(col_func_it, dtypes.items()):
            if dt != np.dtype("O"):
                new_dt[out_col_name] = dt
            elif func == "unique":
                in_dt = col_to_dt[col_name]
                if in_dt == np.dtype("O"):
                    in_dt = pd.StringDtype()
                arrow_dt = pandas_dtype_to_arrow_type(in_dt)
                new_dt[out_col_name] = ArrowDtype(pa.list_(arrow_dt))
            else:
                # do nothing as the result might be string
                new_dt[out_col_name] = dt
        return pd.Series(list(new_dt.values()), index=new_dt.keys())

    def _fill_series_dtype(self, in_data, dtype):
        if len(self.func) != 1 or dtype != np.dtype("O") or in_data.ndim > 1:
            return dtype

        if self.func[0] == "unique":
            in_dt = in_data.dtype
            if in_dt == np.dtype("O"):
                in_dt = pd.StringDtype()
            arrow_dt = pandas_dtype_to_arrow_type(in_dt)
            return ArrowDtype(pa.list_(arrow_dt))
        else:
            return dtype

    def _calc_result_shape(self, df):
        if df.ndim == 2:
            if self.numeric_only:
                df = df.select_dtypes([np.number, np.bool_])
            elif self.bool_only:
                df = df.select_dtypes([np.bool_])

        if self.output_types[0] == OutputType.dataframe:
            test_obj = build_df(df, size=[2, 2], fill_value=[1, 2], ensure_string=True)
        else:
            test_obj = build_series(
                df, size=[2, 2], fill_value=[1, 2], name=df.name, ensure_string=True
            )

        result_df = test_obj.agg(self.raw_func, axis=self.axis, **self.raw_func_kw)
        if isinstance(result_df, pd.DataFrame):
            out_dtypes = self._fill_df_dtypes(df, result_df.dtypes)
        elif isinstance(result_df, pd.Series):
            dtype = self._fill_series_dtype(df, result_df.dtype)
            out_dtypes = pd.Series([dtype], index=[result_df.name])
        else:
            out_dtypes = pd.Series([np.array(result_df).dtype], index=[None])

        if isinstance(result_df, pd.DataFrame):
            self.output_types = [OutputType.dataframe]
            return out_dtypes, result_df.index
        elif isinstance(result_df, pd.Series):
            self.output_types = [OutputType.series]
            return out_dtypes, result_df.index
        else:
            self.output_types = [OutputType.scalar]
            return out_dtypes.iloc[0], None

    def __call__(self, df, output_type=None, dtypes=None, index=None):
        self._output_types = df.op.output_types
        normalize_reduction_funcs(self, ndim=df.ndim)
        compile_reduction_funcs(self, df)
        if output_type is None or dtypes is None:
            with enter_mode(kernel=False, build=False, mock=True):
                dtypes, index = self._calc_result_shape(df)
        else:
            self.output_types = [output_type]

        if self.output_types[0] == OutputType.dataframe:
            if self.axis == 0:
                new_shape = (len(index), len(dtypes))
                new_index = parse_index(index, store_data=True)
            else:
                new_shape = (df.shape[0], len(dtypes))
                new_index = df.index_value
            return self.new_dataframe(
                [df],
                shape=new_shape,
                dtypes=dtypes,
                index_value=new_index,
                columns_value=parse_index(dtypes.index, store_data=True),
            )
        elif self.output_types[0] == OutputType.series:
            if df.ndim == 1:
                new_shape = (len(index),)
                new_index = parse_index(index, store_data=True)
            elif self.axis == 0:
                new_shape = (len(index),)
                new_index = parse_index(index, store_data=True)
            else:
                new_shape = (df.shape[0],)
                new_index = df.index_value
            return self.new_series(
                [df],
                shape=new_shape,
                dtype=dtypes.iloc[0],
                name=dtypes.index[0],
                index_value=new_index,
            )
        elif self.output_types[0] == OutputType.tensor:
            return self.new_tileable([df], dtype=dtypes, shape=(np.nan,))
        else:
            return self.new_scalar([df], dtype=dtypes)


def is_funcs_aggregate(func, func_kw=None, ndim=2):
    func_kw = func_kw or dict()
    if ndim == 1 and func is None:
        func, func_kw = func_kw, dict()

    to_check = []
    if func is not None:
        if isinstance(func, (list, tuple)):
            to_check.extend(func)
        elif isinstance(func, dict):
            if ndim == 2:
                for f in func.values():
                    if (
                        isinstance(f, Iterable)
                        and not isinstance(f, ENTITY_TYPE)
                        and not isinstance(f, str)
                    ):
                        to_check.extend(f)
                    else:
                        to_check.append(f)
            else:
                if any(isinstance(v, tuple) for v in func.values()):
                    raise TypeError("nested renamer is not supported")
                to_check.extend(func.values())
        else:
            to_check.append(func)
    else:
        for v in func_kw.values():
            if (
                not isinstance(v, tuple)
                or len(v) != 2
                or (not isinstance(v[1], str) and not callable(v[1]))
            ):
                raise TypeError("Must provide 'func' or tuples of (column, aggfunc).")
            else:
                to_check.append(v[1])

    compiler = ReductionCompiler()
    for f in to_check:
        if f in _agg_functions:
            continue
        elif callable(f):
            try:
                if ndim == 2:
                    compiler.add_function(f, 2, cols=["A", "B"])
                else:
                    compiler.add_function(f, 1)
            except ValueError:
                return False
        else:
            return False
    return True


def normalize_reduction_funcs(op, ndim=None):
    raw_func = op.raw_func
    if ndim == 1 and raw_func is None:
        raw_func = op.raw_func_kw

    if raw_func is not None:
        if isinstance(raw_func, dict):
            if ndim == 2:
                new_func = OrderedDict()
                for k, v in raw_func.items():
                    if isinstance(v, str) or callable(v):
                        new_func[k] = [v]
                    else:
                        new_func[k] = v
                op.func = new_func
            else:
                op.func = list(raw_func.values())
                op.func_rename = list(raw_func.keys())
        elif (
            isinstance(raw_func, Iterable)
            and not isinstance(raw_func, ENTITY_TYPE)
            and not isinstance(raw_func, str)
        ):
            op.func = list(raw_func)
        else:
            op.func = [raw_func]
    else:
        new_func = OrderedDict()
        new_func_names = OrderedDict()
        for k, v in op.raw_func_kw.items():
            try:
                col_funcs = new_func[v[0]]
                col_func_names = new_func_names[v[0]]
            except KeyError:
                col_funcs = new_func[v[0]] = []
                col_func_names = new_func_names[v[0]] = []
            col_funcs.append(v[1])
            col_func_names.append(k)
        op.func = new_func
        op.func_rename = functools.reduce(
            lambda a, b: a + b, new_func_names.values(), []
        )

    custom_idx = 0
    if isinstance(op.func, list):
        custom_iter = (f for f in op.func if isinstance(f, CustomReduction))
    else:
        custom_iter = (f for f in op.func.values() if isinstance(f, CustomReduction))
    for r in custom_iter:
        if r.name == "<custom>":
            r.name = f"<custom_{custom_idx}>"
            custom_idx += 1


def _add_compiler_functions(
    op: "DataFrameAggregate", compiler: ReductionCompiler, cols=None
):
    if isinstance(op.func, list):
        func_iter = ((None, f) for f in op.func)
        cols_set = set(cols) if cols is not None else None
    else:
        assert cols is not None
        cols_set = set(cols) & set(op.func.keys())
        if len(cols_set) == 0:
            return False
        func_iter = ((col, f) for col, funcs in op.func.items() for f in funcs)

    func_renames = (
        op.func_rename
        if getattr(op, "func_rename", None) is not None
        else itertools.repeat(None)
    )
    for func_rename, (col, f) in zip(func_renames, func_iter):
        if cols_set is not None and col is not None and col not in cols_set:
            continue
        func_name = None
        if isinstance(f, str):
            f, func_name = _agg_functions[f], f
        if func_rename is not None:
            func_name = func_rename
        ndim = 1 if cols is None else 2
        func_cols = [col] if col is not None else None
        compiler.add_function(f, ndim, cols=func_cols, func_name=func_name)
    return True


def compile_reduction_funcs(op: DataFrameAggregate, input: TileableType):
    compiler = ReductionCompiler(axis=getattr(op, "axis", 0))
    cols = input.dtypes.index if input.ndim > 1 else None
    if _add_compiler_functions(op, compiler, cols=cols):
        compiled = compiler.compile()
        op.pre_funcs = compiled.pre_funcs
        op.agg_funcs = compiled.agg_funcs
        op.post_funcs = compiled.post_funcs


def aggregate(df, func=None, axis=0, **kw):
    """
    Aggregate using one or more operations over the specified axis.

    Parameters
    ----------
    df : DataFrame, Series
        Object to aggregate.
    func : list or dict
        Function to use for aggregating the data.
    axis : {0 or ‘index’, 1 or ‘columns’}, default 0
        If 0 or ‘index’: apply function to each column. If 1 or ‘columns’: apply function to each row.
    kw
        Keyword arguments to pass to func.

    Returns
    -------
    scalar, Series or DataFrame
        The return can be:

        * scalar : when Series.agg is called with single function
        * Series : when DataFrame.agg is called with a single function
        * DataFrame : when DataFrame.agg is called with several functions

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([[1, 2, 3],
    ...            [4, 5, 6],
    ...            [7, 8, 9],
    ...            [np.nan, np.nan, np.nan]],
    ...           columns=['A', 'B', 'C']).execute()

    Aggregate these functions over the rows.

    >>> df.agg(['sum', 'min']).execute()
            A     B     C
    min   1.0   2.0   3.0
    sum  12.0  15.0  18.0

    Different aggregations per column.

    >>> df.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']}).execute()
            A    B
    max   NaN  8.0
    min   1.0  2.0
    sum  12.0  NaN

    Aggregate different functions over the columns and rename the index of the resulting DataFrame.

    >>> df.agg(x=('A', 'max'), y=('B', 'min'), z=('C', 'mean')).execute()
         A    B    C
    x  7.0  NaN  NaN
    y  NaN  2.0  NaN
    z  NaN  NaN  6.0

    >>> s = md.Series([1, 2, 3, 4])
    >>> s.agg('min').execute()
    1

    >>> s.agg(['min', 'max']).execute()
    max    4
    min    1
    """
    axis = validate_axis(axis, df)
    use_inf_as_na = kw.pop("_use_inf_as_na", get_pd_option("mode.use_inf_as_na", False))
    if func == "unique":
        # workaround for direct call of unique function which
        #  returns a tensor directly
        func = getattr(df, func)
        if "axis" in inspect.getfullargspec(func).args:
            kw = kw.copy()
            kw["axis"] = axis
        return func(**kw)

    if (
        df.ndim == 2
        and isinstance(func, dict)
        and (df.op.output_types[0] == OutputType.series or axis == 1)
    ):
        raise NotImplementedError(
            "Currently cannot aggregate dicts over axis=1 on %s" % type(df).__name__
        )
    numeric_only = kw.pop("_numeric_only", None)
    bool_only = kw.pop("_bool_only", None)

    output_type = kw.pop("_output_type", None)
    dtypes = kw.pop("_dtypes", None)
    index = kw.pop("_index", None)

    combine_size = kw.pop("_combine_size", None) or kw.pop("combine_size", None)

    if not is_funcs_aggregate(func, func_kw=kw, ndim=df.ndim):
        return df.transform(func, axis=axis, _call_agg=True)

    op = DataFrameAggregate(
        raw_func=copy.deepcopy(func),
        raw_func_kw=copy.deepcopy(kw),
        axis=axis,
        numeric_only=numeric_only,
        bool_only=bool_only,
        combine_size=combine_size,
        use_inf_as_na=use_inf_as_na,
    )

    return op(df, output_type=output_type, dtypes=dtypes, index=index)
