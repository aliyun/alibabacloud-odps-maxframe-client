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

from ....tensor.statistics.bincount import TensorBinCount
from ....tensor.statistics.digitize import TensorDigitize
from ....tensor.statistics.histogram import TensorHistogram, TensorHistogramBinEdges
from ....tensor.statistics.quantile import TensorQuantile
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter

TensorBinCountAdapter = build_method_call_adapter(
    TensorBinCount,
    "bincount",
    0,
    kw_keys=["weights", "minlength"],
    source_module="np",
)
TensorDigitizeAdapter = build_method_call_adapter(
    TensorDigitize,
    "digitize",
    0,
    kw_keys=["bins", "right"],
    source_module="np",
)
TensorHistogramBinEdgesAdapter = build_method_call_adapter(
    TensorHistogramBinEdges,
    "histogram_bin_edges",
    0,
    kw_keys=["bins", "range", "weights"],
    source_module="np",
)
TensorQuantileAdapter = build_method_call_adapter(
    TensorQuantile,
    "quantile",
    0,
    kw_keys=["q", "axis", "overwrite_input", "interpolation", "keepdims"],
    source_module="np",
)


@register_op_adapter(TensorHistogram)
class TensorHistogramAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorHistogram, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")

        inp_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_names = ", ".join(
            context.get_output_tileable_variable(out) for out in op.outputs
        )
        if not op.ret_bins:
            res_var_names += ", _"
        kw_str = self.generate_call_args_with_attributes(
            op, context, kw_keys=["bins", "range", "density", "weights"]
        )
        return [f"{res_var_names} = np.histogram({inp_var_name}, {kw_str})"]
