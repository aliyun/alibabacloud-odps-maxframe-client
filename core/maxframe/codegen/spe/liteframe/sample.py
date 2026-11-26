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

"""SPE adapter for the ``LiteFrameSample`` operator."""

from typing import List

import numpy as np

from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.liteframe.operators.sample import LiteFrameSample


@register_op_adapter(LiteFrameSample)
class LiteFrameSampleAdapter(SPEOperatorAdapter):
    """SPE adapter for ``LiteFrameSample``.

    Single-process execution: uses pandas ``df.sample()`` directly.
    Dispatches across three modes based on per-chunk fields set by the tiler.
    """

    @staticmethod
    def _run_sample(
        df,
        n=None,
        frac=None,
        replace=False,
        weights_col=None,
        seed=None,
        random_state=None,
        chunk_n=None,
        selected_row_ids=None,
    ):
        rs = np.random.RandomState(seed) if seed is not None else random_state
        if selected_row_ids is not None:
            return df.iloc[:0] if not selected_row_ids else df.iloc[selected_row_ids]
        if chunk_n is not None:
            if chunk_n == 0:
                return df.iloc[:0]
            return df.sample(
                n=chunk_n, replace=replace, weights=weights_col, random_state=rs
            )
        # SPE does not tile: chunk_n / selected_row_ids are always None,
        # so fall back to n (API validation makes n and frac mutually
        # exclusive; both None -> pandas default n=1).
        return df.sample(
            n=n,
            frac=frac,
            replace=replace,
            weights=weights_col,
            random_state=rs,
        )

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        res_var = context.get_output_tileable_variable(op.outputs[0])

        args = self._translate_call_args(
            context,
            n=op.n,
            frac=op.frac,
            replace=op.replace,
            weights_col=op.weights_col,
            seed=op.seed,
            random_state=op.random_state,
            chunk_n=op.chunk_n,
            selected_row_ids=op.selected_row_ids,
        )

        return [f"{res_var} = {cls_name}._run_sample({input_var}, {', '.join(args)})"]
