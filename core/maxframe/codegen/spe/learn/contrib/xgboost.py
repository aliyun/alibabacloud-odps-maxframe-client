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

from typing import List

from .....learn.contrib.xgboost.dmatrix import ToDMatrix
from .....learn.contrib.xgboost.predict import XGBPredict
from .....learn.contrib.xgboost.train import XGBTrain
from .....typing_ import OperatorType
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


class _XGBoostAdapter(SPEOperatorAdapter):
    def generate_code(self, op: OperatorType, context: SPECodeContext) -> List[str]:
        context.register_import("xgboost")
        context.register_import("pandas", "pd")
        context.register_import("numpy", "np")


@register_op_adapter(ToDMatrix)
class ToDMatrixAdapter(_XGBoostAdapter):
    def generate_code(self, op: ToDMatrix, context: SPECodeContext) -> List[str]:
        super().generate_code(op, context)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        call_args = (
            "data",
            "label",
            "missing",
            "weight",
            "base_margin",
            "feature_names",
            "feature_types",
            "feature_weights",
            "nthread",
            "group",
            "qid",
            "label_lower_bound",
            "label_upper_bound",
            "enable_categorical",
        )
        args_str = self.generate_call_args_with_attributes(
            op,
            context,
            skip_none=True,
            kw_keys=call_args,
        )
        return [f"{res_var_name} = xgboost.DMatrix({args_str})"]


@register_op_adapter(XGBTrain)
class XGBTrainAdapter(_XGBoostAdapter):
    def generate_code(self, op: XGBTrain, context: SPECodeContext) -> List[str]:
        super().generate_code(op, context)
        context.register_import("pickle")
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        data_var_name = self.translate_var(context, op.dtrain)
        call_args = (
            "params",
            "dtrain",
            "evals",
            "num_boost_round",
            "early_stopping_rounds",
            "obj",
            "feval",
            "callbacks",
            "maximize",
            "verbose_eval",
            "xgb_model",
            "callbacks",
            "custom_metric",
        )
        codes = list()
        args_str = self.generate_call_args_with_attributes(
            op,
            context,
            skip_none=True,
            kw_keys=call_args,
        )
        if op.has_evals_result:
            evals_result_var = context.get_output_tileable_variable(op.outputs[1])
            codes.append(f"{evals_result_var} = dict()")
            args_str = f"{args_str}, evals_result={evals_result_var}"

        expressions = [
            f"{data_var_name}.num_row()",
            f"{data_var_name}.num_col()",
        ]
        codes.extend(
            [
                self.gen_logging_code(
                    context, "Trained data size: (%s, %s)", expressions
                ),
                f"{res_var_name} = xgboost.train({args_str})",
            ]
        )
        return self.gen_timecost_code(context, "Train", codes)


@register_op_adapter(XGBPredict)
class XGBPredictAdapter(_XGBoostAdapter):
    def generate_code(self, op: XGBPredict, context: SPECodeContext) -> List[str]:
        super().generate_code(op, context)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        model_var_name = context.get_input_tileable_variable(op.inputs[1])
        codes = []

        if isinstance(op.inputs[0].op, ToDMatrix):
            data_var = input_var_name
        else:
            data_var = f"{input_var_name}_data"
            codes.append(f"{data_var} = xgboost.DMatrix({input_var_name})")
        # Predict codes
        call_args = (
            "output_margin",
            "pred_leaf",
            "pred_contribs",
            "approx_contribs",
            "pred_interactions",
            "validate_features",
            "training",
            "iteration_range",
            "strict_shape",
        )
        args_str = self.generate_call_args_with_attributes(
            op, context, skip_none=True, kw_keys=call_args
        )
        expressions = [
            f"{data_var}.num_row()",
            f"{data_var}.num_col()",
        ]
        codes.append(
            self.gen_logging_code(
                context, f"Predicted data size: (%s, %s)", expressions
            ),
        )
        codes.append(
            f"{res_var_name} = {model_var_name}.predict({data_var}, {args_str})"
        )

        return self.gen_timecost_code(context, "Predict", codes)
