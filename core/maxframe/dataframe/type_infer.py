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

import dataclasses
import functools
import logging
import warnings
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from ..core import OutputType, get_output_types
from ..errors import TypeInferWarning
from ..utils import make_dtypes, quiet_stdio, unwrap_function
from .utils import build_df, build_empty_df, build_series, make_column_list, parse_index

if TYPE_CHECKING:
    from .core import IndexValue

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class InferredDataFrameMeta:
    output_type: OutputType
    dtypes: Optional[pd.Series] = None
    dtype: Optional[Any] = None
    name: Optional[str] = None
    index_value: Optional["IndexValue"] = None
    maybe_agg: bool = False
    elementwise: bool = False

    def check_absence(self, *args: str) -> None:
        args_set = set(args)
        if self.output_type == OutputType.dataframe:
            args_set.difference_update(["dtype", "name"])
        else:
            args_set.difference_update(["dtypes"])
        absent_args = [arg for arg in sorted(args_set) if getattr(self, arg) is None]
        if absent_args:
            raise TypeError(
                f"Cannot determine {', '.join(absent_args)} by calculating "
                "with mock data, please specify it as arguments"
            )

    def prepend_groupby_keys(self, in_groupby_obj) -> None:
        from .core import GROUPBY_TYPE

        if self.index_value is None:
            return
        assert isinstance(in_groupby_obj, GROUPBY_TYPE)
        res_idx = self.index_value.to_pandas()
        self.index_value = parse_index(
            prepend_group_keys_as_index(res_idx, in_groupby_obj), in_groupby_obj.key
        )


def _get_groupby_input_df(groupby):
    in_df = groupby
    while in_df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
        in_df = in_df.inputs[0]
    return in_df


class MockObjectMixin:
    def __repr__(self):
        return f"<{self.__class__.__name__} for type inference>\n{super().__repr__()}"


class MockDataFrame(MockObjectMixin, pd.DataFrame):
    @property
    def _constructor(self):
        return MockDataFrame


class MockSeries(MockObjectMixin, pd.Series):
    @property
    def _constructor(self):
        return MockSeries


def _wrap_mock(obj):
    if isinstance(obj, (MockDataFrame, MockSeries)):
        return obj
    if isinstance(obj, pd.DataFrame):
        return MockDataFrame(obj)
    if isinstance(obj, pd.Series):
        return MockSeries(obj)
    return obj


def wrap_func_with_mock(func):
    if not callable(func) or isinstance(func, np.ufunc):
        return func

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        wrapped_args = tuple(_wrap_mock(a) for a in args)
        wrapped_kwargs = {k: _wrap_mock(v) for k, v in kwargs.items()}
        return func(*wrapped_args, **wrapped_kwargs)

    return wrapped


def infer_dataframe_return_value(
    df_obj,
    func,
    output_type=None,
    dtypes=None,
    dtype=None,
    name=None,
    index=None,
    inherit_index=False,
    build_kw=None,
    elementwise=None,
    skip_infer=False,
    prepend_index_group_keys=False,
) -> InferredDataFrameMeta:
    from .core import GROUPBY_TYPE, INDEX_TYPE, IndexValue
    from .typing_ import get_function_output_meta

    ret_output_type = None
    ret_index_value = None
    ret_dtypes = dtypes
    maybe_agg = False
    build_kw = build_kw or {}
    obj_key = df_obj.key

    unwrapped_func = unwrap_function(func)

    func_annotation_meta = get_function_output_meta(unwrapped_func, df_obj)
    func_index_value = None
    if func_annotation_meta:
        output_type = output_type or func_annotation_meta.output_type
        dtypes = dtypes if dtypes is not None else func_annotation_meta.dtypes
        dtype = dtype if dtype is not None else func_annotation_meta.dtype
        name = name if name is not None else func_annotation_meta.name
        func_index_value = func_annotation_meta.index_value

    if isinstance(index, INDEX_TYPE):
        index = index.index_value

    if inherit_index:
        index = df_obj.index_value

    if isinstance(index, IndexValue):
        ret_index_value = index
    elif index is not None:
        ret_index_value = parse_index(index, df_obj.key)
    elif func_index_value is not None:
        ret_index_value = func_index_value

    if elementwise is None:
        elementwise = isinstance(unwrapped_func, np.ufunc)
    if elementwise:
        inherit_index = True
        (ret_output_type,) = get_output_types(df_obj)

    if skip_infer or (dtypes is not None or dtype is not None):
        # return directly if skip_infer, or we have at least rough information
        #  about returned dtypes (and return default dtype)
        if ret_index_value is None:
            warnings.warn(
                "Using index from input as result index. Please specify "
                "index parameter if you need other indexes.",
                TypeInferWarning,
            )
            in_df_obj = _get_groupby_input_df(df_obj)
            ret_index_value = parse_index(
                in_df_obj.index_value.to_pandas()[:0], df_obj.key, func
            )
        if output_type is None:
            (output_type,) = get_output_types(df_obj)
            if output_type == OutputType.dataframe_groupby:
                output_type = OutputType.dataframe
            elif output_type == OutputType.series_groupby:
                output_type = OutputType.series

            warnings.warn(
                f"Using source output type {output_type} as result output type. Please "
                "specify output_type parameter if you returns another type.",
                TypeInferWarning,
            )

        ret = InferredDataFrameMeta(
            output_type=output_type,
            dtypes=dtypes,
            dtype=dtype,
            name=name,
            index_value=ret_index_value,
        )
        if prepend_index_group_keys:
            ret.prepend_groupby_keys(df_obj)
        return ret

    warnings.warn(
        f"Inferring return type by calling {func!r}. If you need to avoid this "
        f"invocation, please specify return dtypes and specify skip_infer=True.",
        TypeInferWarning,
    )

    if isinstance(df_obj, GROUPBY_TYPE):
        is_groupby = True
        empty_df_obj = df_obj.op.build_mock_groupby(**build_kw)
    else:
        is_groupby = False
        empty_df_obj = (
            build_df(df_obj, **build_kw)
            if df_obj.ndim == 2
            else build_series(df_obj, **build_kw)
        )

    # Always wrap with Mock for better debugging experience
    empty_df_obj = _wrap_mock(empty_df_obj)

    need_prepend_group_keys = True
    try:
        # Wrap function to convert pandas objects to Mock objects
        wrapped_func = wrap_func_with_mock(func)

        with ExitStack() as stack:
            stack.enter_context(np.errstate(all="ignore"))
            stack.enter_context(quiet_stdio())
            infer_df_obj = wrapped_func(empty_df_obj)

        if ret_index_value is None:
            if (
                infer_df_obj is None
                or not hasattr(infer_df_obj, "index")
                or infer_df_obj.index is None
            ):
                ret_index_value = parse_index(pd.RangeIndex(-1))
            elif (
                infer_df_obj.index is getattr(empty_df_obj, "index", None)
                or inherit_index
            ):
                ret_index_value = df_obj.index_value
            else:
                ret_index_value = parse_index(infer_df_obj.index, obj_key, func)
                need_prepend_group_keys = False

        if isinstance(infer_df_obj, pd.DataFrame):
            if output_type is not None and output_type != OutputType.dataframe:
                raise TypeError(
                    f'Cannot infer output_type as "series", '
                    f'please specify `output_type` as "dataframe"'
                )
            ret_output_type = ret_output_type or OutputType.dataframe
            if ret_dtypes is None:
                ret_dtypes = infer_df_obj.dtypes
        else:
            if output_type is not None and output_type == OutputType.dataframe:
                raise TypeError(
                    f'Cannot infer output_type as "dataframe", '
                    f'please specify `output_type` as "series"'
                )
            ret_output_type = ret_output_type or OutputType.series
            name = name or getattr(infer_df_obj, "name", None)
            dtype = dtype or infer_df_obj.dtype

        if is_groupby and len(infer_df_obj) <= 2:
            # we create mock df with 4 rows, 2 groups
            # if return df has 2 rows, we assume that
            # it's an aggregation operation
            maybe_agg = True

        ret = InferredDataFrameMeta(
            ret_output_type,
            make_dtypes(ret_dtypes),
            make_dtypes(dtype),
            name,
            ret_index_value,
            maybe_agg,
            elementwise=elementwise,
        )
        if prepend_index_group_keys and need_prepend_group_keys:
            ret.prepend_groupby_keys(df_obj)
        return ret
    except:  # noqa: E722  # nosec
        logger.info(
            "Exception raised while inferring meta of function result", exc_info=True
        )
        ret = InferredDataFrameMeta(
            output_type,
            make_dtypes(dtypes),
            make_dtypes(dtype),
            name,
            ret_index_value,
            maybe_agg,
            elementwise=elementwise,
        )
        if prepend_index_group_keys and need_prepend_group_keys:
            ret.prepend_groupby_keys(df_obj)
        return ret


def prepend_group_keys_as_index(res_index, input_groupby) -> pd.Index:
    from .core import GROUPBY_TYPE

    groupby_params = input_groupby.op.groupby_params
    if not groupby_params.get("group_keys"):
        return res_index

    in_obj = input_groupby
    while isinstance(in_obj, GROUPBY_TYPE):
        in_obj = in_obj.inputs[0]

    by_cols = make_column_list(groupby_params.get("by"), in_obj.dtypes) or []
    by_levels = make_column_list(groupby_params.get("level"), in_obj.index.names) or []

    input_df = input_groupby
    while isinstance(input_df, GROUPBY_TYPE):
        input_df = input_df.inputs[0]

    idx_df = res_index.to_frame()
    if not groupby_params.get("as_index"):
        idx_names = [None] + list(res_index.names)
        mock_idx_df = build_empty_df(
            pd.Series([np.dtype(np.int_)], index=[None]), index=idx_df.index
        )
    elif by_cols:
        idx_names = by_cols + list(res_index.names)
        mock_idx_df = build_empty_df(input_df.dtypes[by_cols], index=idx_df.index)
    else:
        idx_names = by_levels + list(res_index.names)
        if len(in_obj.index.names) > 1:
            idx_dtypes = in_obj.index_value.value.dtypes
        else:
            idx_dtypes = pd.Series([in_obj.index.dtype], index=[in_obj.index.name])
        mock_idx_df = build_empty_df(idx_dtypes[by_levels], index=idx_df.index)
    idx_df = pd.concat([mock_idx_df, idx_df], axis=1)
    return pd.MultiIndex.from_frame(idx_df, names=idx_names)
