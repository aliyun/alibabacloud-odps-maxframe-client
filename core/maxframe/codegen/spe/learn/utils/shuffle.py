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

from .....dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from .....learn.utils.shuffle import LearnShuffle
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(LearnShuffle)
class LearnShuffleAdapter(SPEOperatorAdapter):
    def generate_code(self, op: LearnShuffle, context: SPECodeContext) -> List[str]:
        if op.axes == (0,):
            context.register_import(
                "sklearn.utils",
                from_item="shuffle",
                alias="sk_shuffle",
            )
            call_args = [self.translate_var(context, v) for v in op.inputs]
            if op.n_samples:
                call_args.append(
                    "n_samples=" + self.translate_var(context, op.n_samples)
                )
            rs_arg = self.translate_var(context, op.seeds[0])
            res_var_names = [
                context.get_output_tileable_variable(out) for out in op.outputs
            ]
            res_var_names_str = ", ".join(res_var_names)
            return [
                f"{res_var_names[0]}_rs = np.random.RandomState({rs_arg})",
                f"{res_var_names_str} = sk_shuffle({', '.join(call_args)}, "
                f"random_state={res_var_names[0]}_rs)",
            ]
        else:
            lines = []
            out0_var_name = context.get_output_tileable_variable(op.outputs[0])
            if op.n_samples:
                n_samples_str = self.translate_var(context, op.n_samples)
            else:
                n_samples_str = None

            axis_generated = set()
            for axis, seed in zip(op.axes, op.seeds):
                seed_arg = self.translate_var(context, seed)
                lines.append(
                    f"{out0_var_name}_rs{axis} = np.random.RandomState({seed_arg})",
                )
            for idx, inp in enumerate(op.inputs):
                inp_var_name = self.translate_var(context, inp)
                out_var_name = context.get_output_tileable_variable(op.outputs[idx])
                idx_parts = []
                for axis in range(inp.ndim):
                    if axis not in op.axes:
                        idx_parts.append(":")
                    else:
                        if axis not in axis_generated:
                            shape_str = f"{inp_var_name}.shape[{axis}]"
                            if n_samples_str:
                                samples_str = f"min({n_samples_str}, {shape_str})"
                            else:
                                samples_str = shape_str
                            lines.append(
                                f"{out0_var_name}_axis{axis} ="
                                f" {out0_var_name}_rs{axis}.randint("
                                f"0, {shape_str}, {samples_str})"
                            )
                        idx_parts.append(f"{out0_var_name}_axis{axis}")
                idx_str = ", ".join(idx_parts)
                if isinstance(inp, (DATAFRAME_TYPE, SERIES_TYPE)):
                    lines.append(f"{out_var_name} = {inp_var_name}.iloc[{idx_str}]")
                else:
                    lines.append(f"{out_var_name} = {inp_var_name}[{idx_str}]")
            return lines
