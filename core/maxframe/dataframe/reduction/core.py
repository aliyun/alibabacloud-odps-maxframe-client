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
import inspect
from collections import OrderedDict, namedtuple
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import msgpack
import numpy as np
import pandas as pd

from ...core import ENTITY_TYPE, enter_mode, is_build_mode, is_kernel_mode
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DataTypeField,
    DictField,
    Int32Field,
    Serializable,
    StringField,
)
from ...typing_ import TileableType
from ...utils import get_item_if_scalar, get_pd_option, pd_release_version, tokenize
from ..operators import DATAFRAME_TYPE, DataFrameOperator, DataFrameOperatorMixin
from ..utils import (
    build_df,
    build_empty_df,
    build_empty_series,
    build_series,
    parse_index,
    validate_axis,
)

# in pandas<1.3, when aggregating with multiple levels and numeric_only is True,
# object cols not ignored with min-max funcs
_level_reduction_keep_object = pd_release_version[:2] < (1, 3)
# in pandas>=1.3, when dataframes are reduced into series, mixture of float and bool
# results in object.
_reduce_bool_as_object = pd_release_version[:2] != (1, 2)

_idx_reduction_without_numeric_only = pd_release_version[:2] < (1, 5)

NamedAgg = namedtuple("NamedAgg", ["column", "aggfunc"])


class DataFrameReduction(DataFrameOperator):
    _legacy_name = "DataFrameReductionOperator"  # since v2.2.0

    axis = AnyField("axis", default=None)
    skipna = BoolField("skipna", default=True)
    level = AnyField("level", default=None)
    numeric_only = BoolField("numeric_only", default=None)
    bool_only = BoolField("bool_only", default=None)
    min_count = Int32Field("min_count", default=None)
    method = StringField("method", default=None)

    dtype = DataTypeField("dtype", default=None)
    combine_size = Int32Field("combine_size", default=None)
    use_inf_as_na = BoolField("use_inf_as_na", default=None)

    def __init__(self, gpu=None, sparse=None, output_types=None, **kw):
        kw["use_inf_as_na"] = kw.pop(
            "use_inf_as_na", get_pd_option("mode.use_inf_as_na", False)
        )
        super().__init__(gpu=gpu, sparse=sparse, _output_types=output_types, **kw)

    @property
    def is_atomic(self):
        return False

    def get_reduction_args(self, axis=None):
        args = dict(skipna=self.skipna)
        if self.inputs and self.inputs[0].ndim > 1:
            args["axis"] = axis
        if self.numeric_only is not None:
            args["numeric_only"] = self.numeric_only
        if self.bool_only is not None:
            args["bool_only"] = self.bool_only
        return {k: v for k, v in args.items() if v is not None}


# Keep for import compatibility
DataFrameReductionOperator = DataFrameReduction


class DataFrameCumReduction(DataFrameOperator):
    _legacy_name = "DataFrameCumReductionOperator"  # since v2.2.0

    axis = AnyField("axis", default=None)
    skipna = BoolField("skipna", default=None)

    dtype = DataTypeField("dtype", default=None)
    use_inf_as_na = BoolField("use_inf_as_na", default=None)

    def __init__(self, gpu=None, sparse=None, output_types=None, **kw):
        kw["use_inf_as_na"] = kw.pop(
            "use_inf_as_na", get_pd_option("mode.use_inf_as_na", False)
        )
        super().__init__(gpu=gpu, sparse=sparse, _output_types=output_types, **kw)


# Keep for import compatibility
DataFrameCumReductionOperator = DataFrameCumReduction


@functools.lru_cache(100)
def _get_series_reduction_dtype(
    dtype,
    func_name,
    axis=None,
    bool_only=False,
    skipna=True,
    numeric_only=False,
):
    test_series = build_series(dtype=dtype, ensure_string=True)
    if func_name == "count":
        reduced = test_series.count()
    elif func_name == "nunique":
        reduced = test_series.nunique()
    elif func_name in ("all", "any"):
        reduced = getattr(test_series, func_name)(axis=axis, bool_only=bool_only)
    elif func_name == "size":
        reduced = test_series.size
    elif func_name == "str_concat":
        reduced = pd.Series([test_series.str.cat()])
    elif func_name in ("idxmin", "idxmax", "argmin", "argmax"):
        reduced = getattr(test_series, func_name)(axis=axis, skipna=skipna)
    else:
        reduced = getattr(test_series, func_name)(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    return pd.Series(reduced).dtype


@functools.lru_cache(100)
def _get_df_reduction_dtype(
    dtype, func_name, axis=None, bool_only=False, skipna=False, numeric_only=False
):
    test_df = build_series(dtype=dtype, ensure_string=True).to_frame()
    if func_name == "count":
        reduced = getattr(test_df, func_name)(axis=axis, numeric_only=numeric_only)
    elif func_name == "nunique":
        reduced = getattr(test_df, func_name)(axis=axis)
    elif func_name in ("all", "any"):
        reduced = getattr(test_df, func_name)(axis=axis, bool_only=bool_only)
    elif _idx_reduction_without_numeric_only and func_name in ("idxmin", "idxmax"):
        reduced = getattr(test_df, func_name)(axis=axis, skipna=skipna)
    elif func_name == "str_concat":
        reduced = test_df.apply(lambda s: s.str.cat(), axis=axis)
    else:
        reduced = getattr(test_df, func_name)(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    if len(reduced) == 0:
        return None
    return reduced.dtype


class ReductionCallable(Serializable):
    func_name = StringField("func_name")
    kwargs = DictField("kwargs", default=None)

    def __name__(self):
        return self.func_name

    def __call__(self, value):
        kw = self.kwargs.copy()
        if value.ndim == 1:
            kw.pop("bool_only", None)
            kw.pop("numeric_only", None)
            return getattr(value, self.func_name)(**kw)
        else:
            return getattr(value, self.func_name)(**kw)

    def __maxframe_tokenize__(self):
        # make sure compiled functions are correctly cached
        return type(self), self.func_name, self.kwargs


class DataFrameReductionMixin(DataFrameOperatorMixin):
    @classmethod
    def get_reduction_callable(cls, op):
        func_name = getattr(op, "_func_name")
        kw = dict(
            skipna=op.skipna, numeric_only=op.numeric_only, bool_only=op.bool_only
        )
        kw = {k: v for k, v in kw.items() if v is not None}
        return ReductionCallable(func_name=func_name, kwargs=kw)

    def _call_groupby_level(self, df, level):
        return df.groupby(level=level).agg(
            self.get_reduction_callable(self), method=self.method
        )

    def _call_dataframe(self, df):
        axis = getattr(self, "axis", None) or 0
        level = getattr(self, "level", None)
        skipna = getattr(self, "skipna", True)
        numeric_only = getattr(self, "numeric_only", None)
        bool_only = getattr(self, "bool_only", None)
        self.axis = axis = validate_axis(axis, df)
        func_name = getattr(self, "_func_name")

        if level is not None and axis == 1:
            raise NotImplementedError("Not support specify level for axis==1")

        if func_name == "size":
            reduced = pd.Series(
                np.zeros(df.shape[1 - axis]),
                index=df.dtypes.index if axis == 0 else None,
            )
            reduced_cols = list(reduced.index)
            reduced_dtype = reduced.dtype
        elif func_name == "custom_reduction":
            empty_df = build_df(df, ensure_string=True)
            reduced = getattr(self, "custom_reduction").__call_agg__(empty_df)
            reduced_cols = list(reduced.index)
            reduced_dtype = reduced.dtype
        else:
            reduced_cols, dtypes = [], []
            for col, src_dt in df.dtypes.items():
                dt = _get_df_reduction_dtype(
                    src_dt,
                    func_name,
                    axis=axis,
                    bool_only=bool_only,
                    skipna=skipna,
                    numeric_only=numeric_only,
                )
                if dt is not None:
                    reduced_cols.append(col)
                    dtypes.append(dt)
                elif (
                    _level_reduction_keep_object
                    and numeric_only
                    and level is not None
                    and func_name in ("min", "max")
                    and src_dt == np.dtype(object)
                ):  # pragma: no cover
                    reduced_cols.append(col)
                    dtypes.append(np.dtype(object))
            if len(dtypes) == 0:
                reduced_dtype = np.dtype("O")
            elif all(dt == dtypes[0] for dt in dtypes):
                reduced_dtype = dtypes[0]
            else:
                # as we already bypassed dtypes with same values,
                # when has_mixed_bool is True, there are other dtypes
                # other than bool.
                has_mixed_bool = any(dt == np.dtype(bool) for dt in dtypes)
                if _reduce_bool_as_object and has_mixed_bool:
                    reduced_dtype = np.dtype("O")
                elif not all(isinstance(dt, np.dtype) for dt in dtypes):
                    # todo currently we return mixed dtypes as np.dtype('O').
                    #  handle pandas Dtypes in the future more carefully.
                    reduced_dtype = np.dtype("O")
                else:
                    from pandas.core.dtypes.cast import find_common_type

                    reduced_dtype = find_common_type(dtypes)

        if level is not None:
            return self._call_groupby_level(df[reduced_cols], level)

        if axis == 0:
            reduced_shape = (len(reduced_cols),)
            reduced_index_value = parse_index(pd.Index(reduced_cols), store_data=True)
        else:
            reduced_shape = (df.shape[0],)
            reduced_index_value = parse_index(pd.RangeIndex(-1))

        return self.new_series(
            [df],
            shape=reduced_shape,
            dtype=reduced_dtype,
            index_value=reduced_index_value,
        )

    def _call_series(self, series):
        level = getattr(self, "level", None)
        axis = getattr(self, "axis", None)
        skipna = getattr(self, "skipna", True)
        numeric_only = getattr(self, "numeric_only", None)
        bool_only = getattr(self, "bool_only", None)
        self.axis = axis = validate_axis(axis or 0, series)
        func_name = getattr(self, "_func_name")

        if level is not None:
            return self._call_groupby_level(series, level)

        if func_name == "custom_reduction":
            empty_series = build_series(series, ensure_string=True)
            custom_reduction_obj = getattr(self, "custom_reduction")
            result_dtype = getattr(custom_reduction_obj, "result_dtype", None)
            if result_dtype is None:
                result_scalar = custom_reduction_obj.__call_agg__(empty_series)
                if hasattr(result_scalar, "to_pandas"):  # pragma: no cover
                    result_scalar = result_scalar.to_pandas()
                result_dtype = pd.Series(result_scalar).dtype
        else:
            result_dtype = _get_series_reduction_dtype(
                series.dtype,
                func_name,
                axis=axis,
                bool_only=bool_only,
                numeric_only=numeric_only,
                skipna=skipna,
            )
        return self.new_scalar([series], dtype=result_dtype)

    def __call__(self, a):
        if is_kernel_mode() and not getattr(self, "is_atomic", False):
            return self.get_reduction_callable(self)(a)

        if isinstance(a, DATAFRAME_TYPE):
            return self._call_dataframe(a)
        else:
            return self._call_series(a)


class DataFrameCumReductionMixin(DataFrameOperatorMixin):
    def _call_dataframe(self, df):
        axis = getattr(self, "axis", None) or 0
        self.axis = axis = validate_axis(axis, df)

        empty_df = build_empty_df(df.dtypes)
        reduced_df = getattr(empty_df, getattr(self, "_func_name"))(axis=axis)
        return self.new_dataframe(
            [df],
            shape=df.shape,
            dtypes=reduced_df.dtypes,
            index_value=df.index_value,
            columns_value=df.columns_value,
        )

    def _call_series(self, series):
        axis = getattr(self, "axis", None) or 0
        if axis == "index":
            axis = 0
        self.axis = axis

        return self.new_series(
            [series],
            shape=series.shape,
            dtype=series.dtype,
            name=series.name,
            index_value=series.index_value,
        )

    def __call__(self, a):
        if isinstance(a, DATAFRAME_TYPE):
            return self._call_dataframe(a)
        else:
            return self._call_series(a)


class CustomReduction:
    name: Optional[str]
    output_limit: Optional[int]
    kwds: Dict

    # set to True when pre() already performs aggregation
    pre_with_agg = False

    def __init__(self, name=None, is_gpu=None):
        self.name = name or "<custom>"
        self.output_limit = 1
        self._is_gpu = is_gpu

    @property
    def __name__(self):
        return self.name

    @property
    def result_dtype(self):
        return None

    def __call__(self, value):
        if isinstance(value, ENTITY_TYPE):
            from .custom_reduction import build_custom_reduction_result

            return build_custom_reduction_result(value, self)
        return self.__call_agg__(value)

    def __call_agg__(self, value):
        r = self.pre(value)
        if not isinstance(r, tuple):
            r = (r,)
        # update output limit into actual size
        self.output_limit = len(r)

        # only perform aggregation when pre() does not perform aggregation
        if not self.pre_with_agg:
            r = self.agg(*r)
            if not isinstance(r, tuple):
                r = (r,)

        r = self.post(*r)
        return r

    def is_gpu(self):
        return self._is_gpu if not is_build_mode() else False

    def pre(self, value):  # noqa: R0201  # pylint: disable=no-self-use
        return (value,)

    def agg(self, *values):  # noqa: R0201  # pylint: disable=no-self-use
        raise NotImplementedError

    def post(self, *value):  # noqa: R0201  # pylint: disable=no-self-use
        assert len(value) == 1
        return value[0]

    def __maxframe_tokenize__(self):
        import cloudpickle

        return cloudpickle.dumps(self)


class ReductionPreStep(NamedTuple):
    input_key: str
    output_key: str
    columns: Optional[List[str]]
    func_idl: bytes


class ReductionAggStep(NamedTuple):
    input_key: str
    raw_func_name: Optional[str]
    step_func_name: Optional[str]
    map_func_name: Optional[str]
    agg_func_name: Optional[str]
    custom_reduction: Optional[CustomReduction]
    output_key: str
    output_limit: int
    kwds: Dict[str, Any]


class ReductionPostStep(NamedTuple):
    input_keys: List[str]
    output_key: str
    func_name: str
    columns: Optional[List[str]]
    func_idl: bytes
    post_func_aliases: Optional[List[str]] = None


class ReductionSteps(NamedTuple):
    pre_funcs: List[ReductionPreStep]
    agg_funcs: List[ReductionAggStep]
    post_funcs: List[ReductionPostStep]


# lookup table for numpy arithmetic operators in pandas
_func_name_converts = dict(
    greater="gt",
    greater_equal="ge",
    less="lt",
    less_equal="le",
    equal="eq",
    not_equal="ne",
    true_divide="truediv",
    floor_divide="floordiv",
    power="pow",
    subtract="sub",
    multiply="mul",
)
_func_compile_cache = dict()  # type: Dict[str, ReductionSteps]

_idl_primitive_types = (
    type(None),
    int,
    float,
    bool,
    str,
    bytes,
    np.integer,
    np.bool_,
)

IN_VAR_IDL_OP = "in_var"
OUT_VAR_IDL_OP = "out_var"
MASK_VAR_IDL_OP = "mask"
WHERE_VAR_IDL_OP = "where"
LET_VAR_OP = "let"
UNARY_IDL_OP_PREFIX = "unary:"
BINARY_IDL_OP_PREFIX = "bin:"


class ReductionCompiler:
    def __init__(self, axis=0):
        self._axis = axis

        self._key_to_tileable = dict()
        self._output_tileables = []
        self._lambda_counter = 0
        self._custom_counter = 0
        self._func_cache = dict()

        self._compiled_funcs = []
        self._output_key_to_pre_steps = dict()
        self._output_key_to_pre_cols = dict()
        self._output_key_to_agg_steps = dict()
        self._output_key_to_post_steps = dict()
        self._output_key_to_post_cols = dict()
        self._output_key_to_col_func_mapping = dict()

    @classmethod
    def _check_function_valid(cls, func):
        if isinstance(func, functools.partial):
            return cls._check_function_valid(func.func)
        elif not hasattr(func, "__code__"):
            return

        func_code = func.__code__
        func_vars = {n: func.__globals__.get(n) for n in func_code.co_names}
        if func.__closure__:
            func_vars.update(
                {
                    n: cell.cell_contents
                    for n, cell in zip(func_code.co_freevars, func.__closure__)
                }
            )
        # external MaxFrame objects shall not be referenced
        for var_name, val in func_vars.items():
            if isinstance(val, ENTITY_TYPE):
                raise ValueError(
                    f"Variable {var_name} used by {func.__name__} "
                    "cannot be a MaxFrame object"
                )

    @staticmethod
    def _update_col_dict(col_dict: Dict, key: str, cols: List):
        if key in col_dict:
            existing_cols = col_dict[key]
            if existing_cols is not None:
                existing_col_set = set(existing_cols)
                col_dict[key].extend([c for c in cols if c not in existing_col_set])
        else:
            col_dict[key] = list(cols) if cols is not None else None

    def add_function(self, func, ndim, cols=None, func_name=None):
        from .aggregation import _agg_functions

        cols = cols if cols is not None and self._axis == 0 else None

        func_name = func_name or getattr(func, "__name__", None)
        if func_name == "<lambda>" or func_name is None:
            func_name = f"<lambda_{self._lambda_counter}>"
            self._lambda_counter += 1
        if func_name == "<custom>" or func_name is None:
            func_name = f"<custom_{self._custom_counter}>"
            self._custom_counter += 1

        if inspect.isbuiltin(func):
            raw_func_name = getattr(func, "__name__", "N/A")
            if raw_func_name in _agg_functions:
                func = _agg_functions[raw_func_name]
            else:
                raise ValueError(f"Unexpected built-in function {raw_func_name}")

        compile_result = self._compile_function(func, func_name, ndim=ndim)
        self._compiled_funcs.append(compile_result)

        for step in compile_result.pre_funcs:
            self._output_key_to_pre_steps[step.output_key] = step
            self._update_col_dict(self._output_key_to_pre_cols, step.output_key, cols)

        for step in compile_result.agg_funcs:
            self._output_key_to_agg_steps[step.output_key] = step

        for step in compile_result.post_funcs:
            self._output_key_to_post_steps[step.output_key] = step
            self._update_col_dict(self._output_key_to_post_cols, step.output_key, cols)

            if cols is not None:
                col_name_map = (
                    self._output_key_to_col_func_mapping.get(step.output_key) or {}
                )
                for col in cols:
                    col_name_map[col] = func_name
                self._output_key_to_col_func_mapping[step.output_key] = col_name_map

    @staticmethod
    def _build_mock_return_object(func, input_dtype, ndim):
        from ..initializer import DataFrame as MaxDataFrame
        from ..initializer import Series as MaxSeries

        if ndim == 1:
            mock_series = build_empty_series(np.dtype(input_dtype))
            mock_obj = MaxSeries(mock_series)
        else:
            mock_df = build_empty_df(
                pd.Series([np.dtype(input_dtype)] * 2, index=["A", "B"])
            )
            mock_obj = MaxDataFrame(mock_df)

        # calc target tileable to generate DAG
        with enter_mode(kernel=True, build=False):
            return func(mock_obj)

    @enter_mode(build=True)
    def _compile_function(self, func, func_name=None, ndim=1) -> ReductionSteps:
        from ...tensor.arithmetic.core import TensorBinOp, TensorUnaryOp
        from ...tensor.misc import TensorWhere
        from ..arithmetic.core import DataFrameBinOp, DataFrameUnaryOp
        from ..datasource.dataframe import DataFrameDataSource
        from ..datasource.series import SeriesDataSource
        from ..indexing.where import DataFrameWhere
        from .custom_reduction import DataFrameCustomReduction

        func_token = tokenize(func, self._axis, func_name, ndim)
        if func_token in _func_compile_cache:
            return _func_compile_cache[func_token]
        custom_reduction = func if isinstance(func, CustomReduction) else None

        self._check_function_valid(func)

        try:
            func_ret = self._build_mock_return_object(func, float, ndim=ndim)
        except (TypeError, AttributeError):
            # we may encounter lambda x: x.str.cat(...), use an object series to test
            func_ret = self._build_mock_return_object(func, object, ndim=1)
        output_limit = getattr(func, "output_limit", None) or 1

        if not isinstance(func_ret, ENTITY_TYPE):
            raise ValueError(
                f"Custom function should return a MaxFrame object, not {type(func_ret)}"
            )
        if func_ret.ndim >= ndim:
            raise ValueError("Function not a reduction")

        agg_graph = func_ret.build_graph()
        agg_tileables = set(t for t in agg_graph if getattr(t.op, "is_atomic", False))
        # check operators before aggregation
        for t in agg_graph.dfs(
            list(agg_tileables), visit_predicate="all", reverse=True
        ):
            if t not in agg_tileables and not isinstance(
                t.op,
                (
                    DataFrameUnaryOp,
                    DataFrameBinOp,
                    TensorUnaryOp,
                    TensorBinOp,
                    TensorWhere,
                    DataFrameWhere,
                    DataFrameDataSource,
                    SeriesDataSource,
                ),
            ):
                raise ValueError(f"Cannot support operator {type(t.op)} in aggregation")
        # check operators after aggregation
        for t in agg_graph.dfs(list(agg_tileables), visit_predicate="all"):
            if t not in agg_tileables and not isinstance(
                t.op,
                (
                    DataFrameUnaryOp,
                    DataFrameBinOp,
                    TensorWhere,
                    DataFrameWhere,
                    TensorUnaryOp,
                    TensorBinOp,
                ),
            ):
                raise ValueError(f"Cannot support operator {type(t.op)} in aggregation")

        pre_funcs, agg_funcs, post_funcs = [], [], []
        visited_inputs = set()
        # collect aggregations and their inputs
        for t in agg_tileables:
            agg_input_key = t.inputs[0].key

            # collect agg names
            step_func_name = getattr(t.op, "_func_name")
            if step_func_name in ("count", "size"):
                map_func_name, agg_func_name = step_func_name, "sum"
            else:
                map_func_name, agg_func_name = step_func_name, step_func_name

            if isinstance(t.op, DataFrameCustomReduction):
                custom_reduction = custom_reduction or t.op.custom_reduction

            # build agg description
            agg_funcs.append(
                ReductionAggStep(
                    agg_input_key,
                    func_name,
                    step_func_name,
                    map_func_name,
                    agg_func_name,
                    custom_reduction,
                    t.key,
                    output_limit,
                    t.op.get_reduction_args(axis=self._axis),
                )
            )
            # collect agg input and build function
            if agg_input_key not in visited_inputs:
                visited_inputs.add(agg_input_key)
                initial_inputs = list(t.inputs[0].build_graph().iter_indep())
                assert len(initial_inputs) == 1
                input_key = initial_inputs[0].key

                func_idl, _ = self._generate_function_idl(t.inputs[0])
                pre_funcs.append(
                    ReductionPreStep(
                        input_key, agg_input_key, None, msgpack.dumps(func_idl)
                    )
                )
        # collect function output after agg
        func_idl, input_keys = self._generate_function_idl(func_ret)
        post_funcs.append(
            ReductionPostStep(
                input_keys, func_ret.key, func_name, None, msgpack.dumps(func_idl)
            )
        )
        if len(_func_compile_cache) > 100:  # pragma: no cover
            _func_compile_cache.pop(next(iter(_func_compile_cache.keys())))
        result = _func_compile_cache[func_token] = ReductionSteps(
            pre_funcs, agg_funcs, post_funcs
        )
        return result

    def _generate_function_idl(self, out_tileable: TileableType) -> Tuple[List, List]:
        """
        Generate function IDL from tileable DAG

        IDL Format: [
            ["in_var", "input_var_name"],
            ["op", "op_output_var", ["op_arg1", "op_arg2"], {"op_key1": "op_key2"}],
            ["out_var", "output_var_name"],
        ]
        """
        from ...tensor.arithmetic.core import TensorBinOp, TensorUnaryOp
        from ...tensor.datasource import Scalar
        from ...tensor.misc import TensorWhere
        from ..arithmetic.core import DataFrameBinOp, DataFrameUnaryOp
        from ..datasource.dataframe import DataFrameDataSource
        from ..datasource.series import SeriesDataSource
        from ..indexing.where import DataFrameWhere

        input_key_to_var = OrderedDict()
        local_key_to_var = dict()
        idl_lines = []

        input_op_types = (
            DataFrameDataSource,
            SeriesDataSource,
            DataFrameReduction,
        )

        def _gen_expr_str(t):
            # generate code for t
            if t.key in local_key_to_var:
                return

            if isinstance(t.op, input_op_types):
                # tileable is an input arg, build a function variable
                if t.key not in input_key_to_var:  # pragma: no branch
                    input_key_to_var[t.key] = local_key_to_var[
                        t.key
                    ] = f"invar{len(input_key_to_var)}"
            else:
                for inp in t.inputs:
                    _gen_expr_str(inp)

                var_name = local_key_to_var[t.key] = f"var{len(local_key_to_var)}"
                keys_to_vars = {inp.key: local_key_to_var[inp.key] for inp in t.inputs}

                def _interpret_var(v):
                    v = get_item_if_scalar(v)
                    # get representation for variables
                    if hasattr(v, "key"):
                        return keys_to_vars[v.key]
                    elif isinstance(v, _idl_primitive_types):
                        return v
                    else:
                        raise NotImplementedError(
                            f"Type {type(v)} currently not interpretable"
                        )

                func_name = getattr(t.op, "_func_name", None)
                if func_name is None:
                    func_name = getattr(t.op, "_bit_func_name", None)
                # handle function name differences between numpy and pandas arithmetic ops
                if func_name in _func_name_converts:
                    func_name = _func_name_converts[func_name]

                # build given different op types
                if isinstance(t.op, (DataFrameUnaryOp, TensorUnaryOp)):
                    val = _interpret_var(t.inputs[0])
                    statements = [
                        [UNARY_IDL_OP_PREFIX + func_name, var_name, [val], {}]
                    ]
                elif isinstance(t.op, (DataFrameBinOp, TensorBinOp)):
                    lhs, rhs = t.op.lhs, t.op.rhs
                    op_axis = (
                        1 - self._axis
                        if hasattr(lhs, "ndim")
                        and hasattr(rhs, "ndim")
                        and lhs.ndim != rhs.ndim
                        else None
                    )
                    lhs = _interpret_var(lhs)
                    rhs = _interpret_var(rhs)
                    axis_arg = {"axis": op_axis} if op_axis is not None else {}
                    statements = [
                        [
                            BINARY_IDL_OP_PREFIX + func_name,
                            var_name,
                            [lhs, rhs],
                            {},
                            axis_arg,
                        ]
                    ]
                elif isinstance(t.op, TensorWhere):
                    cond = _interpret_var(t.op.condition)
                    x = _interpret_var(t.op.x)
                    y = _interpret_var(t.op.y)
                    statements = [[WHERE_VAR_IDL_OP, var_name, [cond, x, y], {}]]
                elif isinstance(t.op, DataFrameWhere):
                    func_name = (
                        MASK_VAR_IDL_OP if t.op.replace_true else WHERE_VAR_IDL_OP
                    )
                    inp = _interpret_var(t.op.input)
                    cond = _interpret_var(t.op.cond)
                    other = _interpret_var(t.op.other)
                    statements = [
                        [
                            func_name,
                            var_name,
                            [cond, inp, other],
                            {"axis": t.op.axis, "level": t.op.level},
                        ]
                    ]
                elif isinstance(t.op, Scalar):
                    # for scalar inputs of other operators
                    data = _interpret_var(t.op.data)
                    statements = [[LET_VAR_OP, var_name, [data]]]
                else:  # pragma: no cover
                    raise NotImplementedError(
                        f"Does not support aggregating on {type(t.op)}"
                    )

                idl_lines.extend(statements)

        _gen_expr_str(out_tileable)

        input_idls = [
            [IN_VAR_IDL_OP, var_name] for var_name in input_key_to_var.values()
        ]
        output_idls = [[OUT_VAR_IDL_OP, local_key_to_var[out_tileable.key]]]
        return input_idls + idl_lines + output_idls, list(input_key_to_var.keys())

    def compile(self) -> ReductionSteps:
        pre_funcs, agg_funcs, post_funcs = [], [], []
        referred_cols = set()
        for key, step in self._output_key_to_pre_steps.items():
            cols = self._output_key_to_pre_cols[key]
            if cols:
                referred_cols.update(cols)
            pre_funcs.append(
                ReductionPreStep(step.input_key, step.output_key, cols, step.func_idl)
            )

        for step in self._output_key_to_agg_steps.values():
            agg_funcs.append(step)

        for key, step in self._output_key_to_post_steps.items():
            post_cols = self._output_key_to_post_cols[key]
            func_renames = None
            if post_cols:
                col_map = self._output_key_to_col_func_mapping.get(key)
                if col_map:
                    func_renames = [col_map[c] for c in post_cols]

            func_name = step.func_name
            if self._lambda_counter == 1 and step.func_name == "<lambda_0>":
                func_name = "<lambda>"
            if self._custom_counter == 1 and step.func_name == "<custom_0>":
                func_name = "<custom>"

            post_funcs.append(
                ReductionPostStep(
                    step.input_keys,
                    step.output_key,
                    func_name,
                    post_cols,
                    step.func_idl,
                    func_renames,
                )
            )

        return ReductionSteps(pre_funcs, agg_funcs, post_funcs)
