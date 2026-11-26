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

import inspect
from typing import List, Optional

from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.learn.contrib.lightgbm._predict import LGBMPredict
from maxframe.learn.contrib.lightgbm._train import LGBMTrain
from maxframe.learn.contrib.lightgbm.dataset import ToLGBMDataset
from maxframe.lib.version import parse as parse_version
from maxframe.typing_ import OperatorType


class _LightGBMAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: OperatorType, context: SPECodeContext
    ) -> Optional[List[str]]:
        context.register_import("lightgbm")
        context.register_import("pandas", "pd")
        context.register_import("numpy", "np")

    @classmethod
    def append_categorical_feature(cls, op, context, args_str: str) -> str:
        # emit even when None: lightgbm reads None as "no categorical feature",
        # so dropping it would silently fall back to "auto"
        arg = "categorical_feature=" + cls.translate_var(
            context, op.categorical_feature
        )
        return f"{args_str}, {arg}" if args_str else arg

    @classmethod
    def append_feature_name(cls, op, context, args_str: str) -> str:
        # None is "let the Dataset decide", so it stays omitted; "auto" and an
        # explicit list are real values and must be forwarded
        if op.feature_name is None:
            return args_str
        arg = "feature_name=" + cls.translate_var(context, op.feature_name)
        return f"{args_str}, {arg}" if args_str else arg

    @classmethod
    def _train_accepts_categorical_feature(cls) -> bool:
        # >= 4.6 dropped feature_name and categorical_feature from train(); the
        # Dataset values are then authoritative (same gate as the DPE executor)
        from maxframe.learn.contrib.lightgbm import core as lgbm_core

        lgbm_mod = lgbm_core.lightgbm  # None when lightgbm is not installed
        if lgbm_mod is None:
            return True
        return parse_version(lgbm_mod.__version__).release[:2] < (4, 6)


@register_op_adapter(ToLGBMDataset)
class ToLGBMDatasetAdapter(_LightGBMAdapter):
    def generate_code(self, op: ToLGBMDataset, context: SPECodeContext) -> List[str]:
        super().generate_code(op, context)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        call_args = (
            "data",
            "label",
            "reference",
            "weight",
            "init_score",
            "group",
            "feature_name",
            "params",
            "free_raw_data",
            "position",
        )
        args_str = self.generate_call_args_with_attributes(
            op,
            context,
            skip_none=True,
            kw_keys=call_args,
        )
        args_str = self.append_categorical_feature(op, context, args_str)
        return [f"{res_var_name} = lightgbm.Dataset({args_str})"]


@register_op_adapter(LGBMTrain)
class LGBMTrainAdapter(_LightGBMAdapter):
    @classmethod
    def _run_lgbm_train(cls, **kwargs):
        import lightgbm

        if "evals_result" in kwargs:
            evals_result = kwargs.pop("evals_result")
            if "evals_result" in inspect.getfullargspec(lightgbm.train).args:
                kwargs["evals_result"] = evals_result
            else:
                # lightgbm >= 4.0 moves evals_result recording into callbacks
                from lightgbm.callback import record_evaluation

                callbacks = list(kwargs.get("callback") or [])
                callbacks.append(record_evaluation(evals_result))
                kwargs["callbacks"] = callbacks
        return lightgbm.train(**kwargs)

    def generate_code(self, op: LGBMTrain, context: SPECodeContext) -> List[str]:
        super().generate_code(op, context)
        context.register_import("pickle")
        context.register_import("lightgbm.callback", from_item="record_evaluation")
        context.register_import(__name__, from_item=type(self).__name__)

        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        data_var_name = self.translate_var(context, op.train_set)
        call_kwargs = (
            "params",
            "train_set",
            "num_boost_round",
            "valid_sets",
            "valid_names",
            "feval",
            "init_model",
            "keep_training_booster",
            "tree_learner",
            "callbacks",
        )
        codes = list()
        args_str = self.generate_call_args_with_attributes(
            op,
            context,
            skip_none=True,
            kw_keys=call_kwargs,
        )
        if self._train_accepts_categorical_feature():
            args_str = self.append_feature_name(op, context, args_str)
            args_str = self.append_categorical_feature(op, context, args_str)
        if op.has_evals_result:
            evals_result_var = context.get_output_tileable_variable(op.outputs[1])
            codes.extend(
                [
                    f"{evals_result_var} = dict()",
                ]
            )
            args_str = f"{args_str}, evals_result={evals_result_var}"

        expressions = [
            f"{data_var_name}.num_data()",
            f"{data_var_name}.num_feature()",
        ]
        codes.extend(
            [
                self.gen_logging_code(
                    context, "Trained data size: (%s, %s)", expressions
                ),
                f"{res_var_name} = {type(self).__name__}._run_lgbm_train({args_str})",
            ]
        )
        return self.gen_timecost_code(context, "Train", codes)


@register_op_adapter(LGBMPredict)
class LGBMPredictAdapter(_LightGBMAdapter):
    def generate_code(self, op: LGBMPredict, context: SPECodeContext) -> List[str]:
        super().generate_code(op, context)
        res_var_names = [
            context.get_output_tileable_variable(out) for out in op.outputs
        ]
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        model_var_name = context.get_input_tileable_variable(op.inputs[1])
        codes = []

        # Predict codes
        call_args = (
            "start_iteration",
            "num_iterations",
            "raw_score",
            "pred_leaf",
            "pred_contrib",
        )
        args_str = self.generate_call_args_with_attributes(
            op, context, skip_none=True, kw_keys=call_args, **op.kwds
        )
        codes.append(
            f"{', '.join(res_var_names)} = {model_var_name}.predict("
            f"{input_var_name}, {args_str})"
        )

        return self.gen_timecost_code(context, "Predict", codes)
