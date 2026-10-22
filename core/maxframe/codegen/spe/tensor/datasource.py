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

from typing import List, Union

from maxframe.codegen import EngineAcceptance
from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.codegen.spe.liteframe.expressions import _materialize_range_columns
from maxframe.codegen.spe.utils import build_method_call_adapter
from maxframe.tensor.datasource import (
    Scalar,
    TensorArange,
    TensorDiag,
    TensorEmpty,
    TensorEye,
    TensorLinspace,
    TensorOnesLike,
    TensorTril,
    TensorZeros,
    TensorZerosLike,
)
from maxframe.tensor.datasource.array import ArrayDataSource
from maxframe.tensor.datasource.from_dataframe import TensorFromDataFrame
from maxframe.tensor.datasource.from_liteframe import TensorFromLiteFrame
from maxframe.tensor.datasource.full import TensorFull
from maxframe.tensor.datasource.ones import TensorOnes
from maxframe.tensor.datasource.tri_array import TensorTriArray

TensorArangeAdapter = build_method_call_adapter(
    TensorArange,
    "arange",
    "start",
    "stop",
    "step",
    source_module="np",
    kw_keys=["dtype"],
)


@register_op_adapter(ArrayDataSource)
class ArrayDataSourceAdapter(SPEOperatorAdapter):
    def generate_code(self, op: ArrayDataSource, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        data_var = context.register_operator_constants(op.data)
        dtype_var = context.register_operator_constants(op.outputs[0].dtype)
        return [
            f"{res_var_name} = np.asarray(({data_var}), "
            f"dtype={dtype_var}, order={op.order!r})"
        ]


TensorDiagAdapter = build_method_call_adapter(
    TensorDiag, "diag", "v", "k", source_module="np"
)


@register_op_adapter(TensorEmpty)
class TensorEmptyAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorEmpty, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")

        shape_str = self.translate_var(context, op.outputs[0].shape)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        dtype_var = context.register_operator_constants(op.outputs[0].dtype)
        return [
            f"{res_var_name} = np.empty({shape_str}, "
            f"dtype={dtype_var}, order={op.order!r})"
        ]


TensorEyeAdapter = build_method_call_adapter(
    TensorEye, "eye", "N", "M", kw_keys=["k", "dtype", "order"], source_module="np"
)


@register_op_adapter(TensorFull)
class TensorFullAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorFull, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")

        shape_str = self.translate_var(context, op.outputs[0].shape)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        kws = self.generate_call_args_with_attributes(
            op, context, "fill_value", kw_keys=["dtype", "order"]
        )
        return [f"{res_var_name} = np.full({shape_str}, {kws})"]


TensorFromDataFrameAdapter = build_method_call_adapter(TensorFromDataFrame, "to_numpy")


@register_op_adapter(TensorFromLiteFrame)
class TensorFromLiteFrameAdapter(SPEOperatorAdapter):
    """SPE adapter for ``TensorFromLiteFrame``.

    Mirrors the DPE executor: virtual range columns are materialized into
    physical Int64 columns, the visible columns are selected in order, and
    the result is converted to a numpy array cast to ``dtype`` in F order.
    """

    @staticmethod
    def _from_liteframe(df, visible_columns, dtype, range_columns):
        work_df = df
        if range_columns:
            work_df = _materialize_range_columns(work_df, range_columns)
        if visible_columns is not None and list(work_df.columns) != list(
            visible_columns
        ):
            work_df = work_df[list(visible_columns)]
        return work_df.to_numpy().astype(dtype, order="F")

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        in_tileable = op.inputs[0]
        range_columns = None
        metadata = getattr(in_tileable, "frame_metadata", None)
        if metadata and metadata.range_columns:
            range_columns = {
                name: (info.start, info.stop, info.step)
                for name, info in metadata.range_columns.items()
            }

        vis_var = context.register_operator_constants(op.visible_columns)
        dtype_var = context.register_operator_constants(op.outputs[0].dtype)
        range_cols_var = context.register_operator_constants(range_columns)
        return [
            f"{res_var_name} = {cls_name}._from_liteframe("
            f"{input_var}, {vis_var}, {dtype_var}, {range_cols_var})"
        ]


TensorLinspaceAdapter = build_method_call_adapter(
    TensorLinspace,
    "linspace",
    "start",
    "stop",
    kw_keys=["num", "endpoint", "order"],
    source_module="np",
)


@register_op_adapter([TensorOnes, TensorZeros])
class TensorOnesZerosAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: Union[TensorOnes, TensorZeros], context: SPECodeContext
    ) -> List[str]:
        context.register_import("numpy", "np")

        shape_str = self.translate_var(context, op.shape)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        dtype_var = context.register_operator_constants(op.outputs[0].dtype)
        func_name = "ones" if isinstance(op, TensorOnes) else "zeros"
        return [
            f"{res_var_name} = np.{func_name}({shape_str}, "
            f"dtype={dtype_var}, order={op.order!r})"
        ]


@register_op_adapter([TensorOnesLike, TensorZerosLike])
class TensorOnesZerosLikeAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: Union[TensorOnesLike, TensorZerosLike], context: SPECodeContext
    ) -> List[str]:
        context.register_import("numpy", "np")

        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        shape_str = self.translate_var(context, op.outputs[0].shape)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        dtype_var = context.register_operator_constants(op.outputs[0].dtype)
        func_name = "ones_like" if isinstance(op, TensorOnesLike) else "zeros_like"
        return [
            f"{res_var_name} = np.{func_name}({input_var_name}, "
            f"dtype={dtype_var}, order={op.order!r}, shape={shape_str})"
        ]


@register_op_adapter(Scalar)
class TensorScalarAdapter(SPEOperatorAdapter):
    def accepts(self, op: Scalar) -> EngineAcceptance:
        return EngineAcceptance.SUCCESSOR

    def generate_code(self, op: Scalar, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        vals = self.generate_call_args_with_attributes(op, context, "data")
        return [f"{res_var_name} = np.asarray({vals})"]


@register_op_adapter(TensorTriArray)
class TensorTriArrayAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorTriArray, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")

        inp_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        kws = self.generate_call_args_with_attributes(op, context, kw_keys=["k"])
        func_name = "tril" if isinstance(op, TensorTril) else "triu"
        return [f"{res_var_name} = np.{func_name}({inp_var_name}, {kws})"]
