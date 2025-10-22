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

from typing import Any, Dict, List, Union

from ....dataframe.reduction import (
    DataFrameAggregate,
    DataFrameAll,
    DataFrameAny,
    DataFrameArgMax,
    DataFrameArgMin,
    DataFrameCount,
    DataFrameIdxMax,
    DataFrameIdxMin,
    DataFrameKurtosis,
    DataFrameMax,
    DataFrameMean,
    DataFrameMedian,
    DataFrameMin,
    DataFrameMode,
    DataFrameNunique,
    DataFrameProd,
    DataFrameSem,
    DataFrameSkew,
    DataFrameSum,
    DataFrameUnique,
    DataFrameVar,
)
from ....dataframe.reduction.core import DataFrameCumReduction, DataFrameReduction
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(DataFrameCumReduction)
class DataFrameCumsumAdapter(SPEOperatorAdapter):
    """
    TODO: Refine this in window functions
    """

    def generate_code(
        self, op: DataFrameCumReduction, context: SPECodeContext
    ) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        args = []
        if op.axis is not None:
            args.append(f"axis={op.axis!r}")
        if op.skipna is not None:
            args.append(f"skipna={op.skipna!r}")
        args_str = ", ".join(args)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = {input_var_name}.{op._func_name}({args_str})"]


@register_op_adapter(DataFrameAggregate)
class DataFrameAggregateAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameAggregate, context: SPECodeContext):
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        agg_func_str = self.translate_var(context, op.raw_func)
        axis_str = self.translate_var(context, op.axis)

        return [
            f"{res_var_name} = {input_var_name}.agg({agg_func_str}, axis={axis_str})"
        ]


# TODO: DataFrameStrConcat, DataFrameReductionSize and DataFrameCustomReduction
@register_op_adapter(
    [
        DataFrameAll,
        DataFrameAny,
        DataFrameArgMax,
        DataFrameArgMin,
        DataFrameCount,
        DataFrameIdxMax,
        DataFrameIdxMin,
        DataFrameMax,
        DataFrameMean,
        DataFrameMedian,
        DataFrameMin,
        DataFrameProd,
        DataFrameSum,
    ]
)
class DataFrameReductionAdapter(SPEOperatorAdapter):
    _common_args = ["axis", "skipna", "numeric_only", "bool_only", "level", "min_count"]

    def extra_args(self, op: DataFrameReduction) -> Dict[str, Any]:
        """
        Get the extra arguments of the API call.

        Parameters
        ----------
        op : DataFrameReduction
            The DataFrameReductionOperator instance.

        Returns
        -------
        Dict[str, Any]:
            Extra arguments key and values.
        """
        return dict()

    def generate_code(
        self, op: DataFrameReduction, context: SPECodeContext
    ) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        kwargs = dict()
        for k in self._common_args:
            v = getattr(op, k)
            if v is not None:
                kwargs[k] = v
        kwargs.update(self.extra_args(op))
        args_str = ", ".join(self._translate_call_args(context, **kwargs))
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = {input_var_name}.{op._func_name}({args_str})"]


@register_op_adapter(DataFrameKurtosis)
class DataFrameKurtosisAdapter(DataFrameReductionAdapter):
    def extra_args(self, op: DataFrameKurtosis) -> Dict[str, Any]:
        return {"bias": op.bias, "fisher": op.fisher}


@register_op_adapter(DataFrameNunique)
class DataFrameNuniqueAdapter(DataFrameReductionAdapter):
    _common_args = ["level", "min_count"]

    def extra_args(self, op: DataFrameNunique) -> Dict[str, Any]:
        if op.inputs[0].ndim == 2:
            return {"axis": op.axis, "dropna": op.dropna}
        return {"dropna": op.dropna}


@register_op_adapter(
    [
        DataFrameSem,
        DataFrameSkew,
        DataFrameVar,
    ]
)
class DataFrameVarAdapter(DataFrameReductionAdapter):
    def extra_args(
        self, op: Union[DataFrameSem, DataFrameSkew, DataFrameVar]
    ) -> Dict[str, Any]:
        return {"ddof": op.ddof}


@register_op_adapter(DataFrameUnique)
class DataFrameUniqueAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameUnique, context: SPECodeContext) -> List[str]:
        context.register_import("pandas", "pd")
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = pd.unique({input_var_name})"]


@register_op_adapter(DataFrameMode)
class DataFrameModeAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameMode, context: SPECodeContext) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        args = []
        if op.inputs[0].ndim == 2:
            if op.axis is not None:
                args.append(f"axis={op.axis!r}")
            if op.numeric_only is not None:
                args.append(f"numeric_only={op.numeric_only!r}")
        if op.dropna is not None:
            args.append(f"dropna={op.dropna!r}")
        args_str = ", ".join(args)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = {input_var_name}.mode({args_str})"]
