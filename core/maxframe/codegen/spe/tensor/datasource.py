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

from typing import List, Union

from ....tensor.datasource import (
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
from ....tensor.datasource.array import ArrayDataSource
from ....tensor.datasource.from_dataframe import TensorFromDataFrame
from ....tensor.datasource.full import TensorFull
from ....tensor.datasource.ones import TensorOnes
from ....tensor.datasource.tri_array import TensorTriArray
from ... import EngineAcceptance
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter

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
