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
from typing import List, MutableMapping, Union

import numpy as np
import pandas as pd

from ...core import ENTITY_TYPE, EntityData
from ...serialization.serializables import AnyField
from ...tensor.core import TENSOR_TYPE
from ...utils import classproperty, make_dtype
from ..core import DATAFRAME_TYPE, SERIES_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..ufunc.tensor import TensorUfuncMixin
from ..utils import (
    build_empty_df,
    infer_dtype,
    infer_dtypes,
    infer_index_value,
    parse_index,
    validate_axis,
)


class DataFrameBinOpMixin(DataFrameOperatorMixin):
    @classproperty
    def _operator(self):
        raise NotImplementedError

    @classmethod
    def _calc_properties(cls, x1, x2=None, axis="columns", level=None):
        if isinstance(x1, DATAFRAME_TYPE) and (
            x2 is None or pd.api.types.is_scalar(x2) or isinstance(x2, TENSOR_TYPE)
        ):
            if pd.api.types.is_scalar(x2):
                dtypes = cls._operator(build_empty_df(x1.dtypes), x2).dtypes
            elif x1.dtypes is not None and isinstance(x2, TENSOR_TYPE):
                dtypes = pd.Series(
                    [infer_dtype(dt, x2.dtype, cls._operator) for dt in x1.dtypes],
                    index=x1.dtypes.index,
                )
            else:  # pragma: no cover
                dtypes = x1.dtypes
            return {
                "shape": x1.shape,
                "dtypes": dtypes,
                "columns_value": x1.columns_value,
                "index_value": x1.index_value,
            }

        if isinstance(x1, SERIES_TYPE) and (
            x2 is None or pd.api.types.is_scalar(x2) or isinstance(x2, TENSOR_TYPE)
        ):
            x2_dtype = x2.dtype if hasattr(x2, "dtype") else type(x2)
            x2_dtype = make_dtype(x2_dtype)
            if hasattr(cls, "return_dtype"):
                dtype = cls.return_dtype
            else:
                dtype = infer_dtype(x1.dtype, x2_dtype, cls._operator)
            ret = {"shape": x1.shape, "dtype": dtype}
            if pd.api.types.is_scalar(x2) or (
                hasattr(x2, "ndim") and (x2.ndim == 0 or x2.ndim == 1)
            ):
                ret["name"] = x1.name
            ret["index_value"] = x1.index_value
            return ret

        if isinstance(x1, DATAFRAME_TYPE) and isinstance(x2, DATAFRAME_TYPE):
            index_shape, column_shape, dtypes, columns, index = (
                np.nan,
                np.nan,
                None,
                None,
                None,
            )

            if (
                x1.columns_value is not None
                and x2.columns_value is not None
                and x1.columns_value.key == x2.columns_value.key
            ):
                dtypes = pd.Series(
                    [
                        infer_dtype(dt1, dt2, cls._operator)
                        for dt1, dt2 in zip(x1.dtypes, x2.dtypes)
                    ],
                    index=x1.dtypes.index,
                )
                columns = copy.copy(x1.columns_value)
                column_shape = len(dtypes)
            elif x1.dtypes is not None and x2.dtypes is not None:
                dtypes = infer_dtypes(x1.dtypes, x2.dtypes, cls._operator)
                columns = parse_index(dtypes.index, store_data=True)
                column_shape = len(dtypes)
            if x1.index_value is not None and x2.index_value is not None:
                if x1.index_value.key == x2.index_value.key:
                    index = copy.copy(x1.index_value)
                    index_shape = x1.shape[0]
                else:
                    index = infer_index_value(
                        x1.index_value, x2.index_value, level=level
                    )
                    if index.key == x1.index_value.key == x2.index_value.key and (
                        not np.isnan(x1.shape[0]) or not np.isnan(x2.shape[0])
                    ):
                        index_shape = (
                            x1.shape[0] if not np.isnan(x1.shape[0]) else x2.shape[0]
                        )

            return {
                "shape": (index_shape, column_shape),
                "dtypes": dtypes,
                "columns_value": columns,
                "index_value": index,
            }

        if isinstance(x1, DATAFRAME_TYPE) and isinstance(x2, SERIES_TYPE):
            if axis == "columns" or axis == 1:
                index_shape = x1.shape[0]
                index = x1.index_value
                column_shape, dtypes, columns = np.nan, None, None
                if x1.columns_value is not None and x1.index_value is not None:
                    if x1.columns_value.key == x2.index_value.key:
                        dtypes = pd.Series(
                            [
                                infer_dtype(dt, x2.dtype, cls._operator)
                                for dt in x1.dtypes
                            ],
                            index=x1.dtypes.index,
                        )
                        columns = copy.copy(x1.columns_value)
                        column_shape = len(dtypes)
                    else:  # pragma: no cover
                        dtypes = x1.dtypes  # FIXME
                        columns = infer_index_value(
                            x1.columns_value, x2.index_value, level=level
                        )
                        column_shape = np.nan
            else:
                assert axis == "index" or axis == 0
                column_shape = x1.shape[1]
                columns = x1.columns_value
                dtypes = x1.dtypes
                index_shape, index = np.nan, None
                if x1.index_value is not None and x2.index_value is not None:
                    if x1.index_value.key == x2.index_value.key:
                        dtypes = pd.Series(
                            [
                                infer_dtype(dt, x2.dtype, cls._operator)
                                for dt in x1.dtypes
                            ],
                            index=x1.dtypes.index,
                        )
                        index = copy.copy(x1.index_value)
                        index_shape = x1.shape[0]
                    else:
                        if x1.dtypes is not None:
                            dtypes = pd.Series(
                                [
                                    infer_dtype(dt, x2.dtype, cls._operator)
                                    for dt in x1.dtypes
                                ],
                                index=x1.dtypes.index,
                            )
                        index = infer_index_value(
                            x1.index_value, x2.index_value, level=level
                        )
                        index_shape = np.nan
            return {
                "shape": (index_shape, column_shape),
                "dtypes": dtypes,
                "columns_value": columns,
                "index_value": index,
            }

        if isinstance(x1, SERIES_TYPE) and isinstance(x2, SERIES_TYPE):
            index_shape, dtype, index = np.nan, None, None

            dtype = infer_dtype(x1.dtype, x2.dtype, cls._operator)
            if x1.index_value is not None and x2.index_value is not None:
                if x1.index_value.key == x2.index_value.key:
                    index = copy.copy(x1.index_value)
                    index_shape = x1.shape[0]
                else:
                    index = infer_index_value(
                        x1.index_value, x2.index_value, level=level
                    )
                    if index.key == x1.index_value.key == x2.index_value.key and (
                        not np.isnan(x1.shape[0]) or not np.isnan(x2.shape[0])
                    ):
                        index_shape = (
                            x1.shape[0] if not np.isnan(x1.shape[0]) else x2.shape[0]
                        )

            ret = {"shape": (index_shape,), "dtype": dtype, "index_value": index}
            if x1.name == x2.name:
                ret["name"] = x1.name
            return ret

        raise NotImplementedError("Unknown combination of parameters")

    def _check_inputs(self, x1, x2):
        if isinstance(x1, TENSOR_TYPE) or isinstance(x2, TENSOR_TYPE):
            tensor, other = (x1, x2) if isinstance(x1, TENSOR_TYPE) else (x2, x1)
            if isinstance(other, DATAFRAME_TYPE):
                if self.axis == "index" or self.axis == 0:
                    other_shape = tuple(reversed(other.shape))
                else:
                    other_shape = other.shape
                if tensor.ndim == 2 and tensor.shape != other_shape:
                    raise ValueError(
                        f"Unable to coerce to DataFrame, shape must be {other_shape}: "
                        f"given {tensor.shape}"
                    )
                elif tensor.ndim == 1 and tensor.shape[0] != other_shape[1]:
                    raise ValueError(
                        f"Unable to coerce to Series, length must be {other_shape[1]}: "
                        f"given {tensor.shape[0]}"
                    )
                elif tensor.ndim > 2:
                    raise ValueError(
                        "Unable to coerce to Series/DataFrame, dim must be <= 2"
                    )
            if isinstance(other, SERIES_TYPE):
                if tensor.ndim == 1 and (tensor.shape[0] != other.shape[0]):
                    raise ValueError(
                        f"Unable to coerce to Series, length must be {other.shape[0]}: "
                        f"given {tensor.shape[0]}"
                    )
                elif tensor.ndim > 1:
                    raise ValueError("Unable to coerce to Series, dim must be 1")

    def _call(self, x1, x2):
        self._check_inputs(x1, x2)
        if isinstance(x1, DATAFRAME_TYPE) or isinstance(x2, DATAFRAME_TYPE):
            df1, df2 = (x1, x2) if isinstance(x1, DATAFRAME_TYPE) else (x2, x1)
            kw = self._calc_properties(df1, df2, axis=self.axis, level=self.level)
            if not pd.api.types.is_scalar(df2):
                return self.new_dataframe([x1, x2], **kw)
            else:
                return self.new_dataframe([df1], **kw)
        if isinstance(x1, SERIES_TYPE) or isinstance(x2, SERIES_TYPE):
            s1, s2 = (x1, x2) if isinstance(x1, SERIES_TYPE) else (x2, x1)
            kw = self._calc_properties(s1, s2, level=self.level)
            if not pd.api.types.is_scalar(s2):
                return self.new_series([x1, x2], **kw)
            else:
                return self.new_series([s1], **kw)
        raise NotImplementedError(
            "Only support add dataframe, series or scalar for now"
        )

    def __call__(self, x1, x2):
        x1 = self._process_input(x1)
        x2 = self._process_input(x2)
        if isinstance(x1, SERIES_TYPE) and isinstance(x2, DATAFRAME_TYPE):
            # reject invoking series's op on dataframe
            raise NotImplementedError
        return self._call(x1, x2)

    def rcall(self, x1, x2):
        x1 = self._process_input(x1)
        x2 = self._process_input(x2)
        if isinstance(x1, SERIES_TYPE) and isinstance(x2, DATAFRAME_TYPE):
            # reject invoking series's op on dataframe
            raise NotImplementedError
        return self._call(x2, x1)

    @classmethod
    def estimate_size(
        cls, ctx: MutableMapping[str, Union[int, float]], op: "DataFrameOperator"
    ):
        ctx[op.outputs[0].key] = max(ctx[inp.key] for inp in (op.inputs or ()))


class DataFrameBinOp(DataFrameBinOpMixin, DataFrameOperator):
    axis = AnyField("axis", default=None)
    level = AnyField("level", default=None)
    fill_value = AnyField("fill_value", default=None)
    lhs = AnyField("lhs")
    rhs = AnyField("rhs")

    def __init__(self, output_types=None, axis=0, **kw):
        axis = validate_axis(axis)
        super().__init__(_output_types=output_types, axis=axis, **kw)

    @classmethod
    def _set_inputs(cls, op: "DataFrameBinOp", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if len(op._inputs) == 2:
            op.lhs = op._inputs[0]
            op.rhs = op._inputs[1]
        else:
            if isinstance(op.lhs, ENTITY_TYPE):
                op.lhs = op._inputs[0]
            elif isinstance(op.rhs, ENTITY_TYPE):
                op.rhs = op._inputs[0]


class DataFrameUnaryOpMixin(DataFrameOperatorMixin):
    __slots__ = ()


class DataFrameUnaryOp(DataFrameUnaryOpMixin, DataFrameOperator):
    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @classmethod
    def _get_output_dtype(cls, df):
        if df.ndim == 2:
            return df.dtypes
        else:
            return df.dtype

    def __call__(self, df):
        self.output_types = df.op.output_types
        if df.ndim == 2:
            return self.new_dataframe(
                [df],
                shape=df.shape,
                dtypes=self._get_output_dtype(df),
                columns_value=df.columns_value,
                index_value=df.index_value,
            )
        else:
            series = df
            return self.new_series(
                [series],
                shape=series.shape,
                name=series.name,
                index_value=series.index_value,
                dtype=self._get_output_dtype(series),
            )


class DataFrameArithmeticTreeMixin:
    @classmethod
    def _set_inputs(cls, op: "DataFrameOperator", inputs: List[EntityData]):
        inputs = op._get_inputs_data(inputs)
        setattr(op, "_inputs", inputs)


class DataFrameUnaryUfunc(DataFrameUnaryOp, TensorUfuncMixin):
    pass


class DataFrameBinopUfunc(DataFrameBinOp, TensorUfuncMixin):
    pass
