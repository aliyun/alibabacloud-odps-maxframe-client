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

import pandas as pd

from maxframe.codegen.spe.liteframe.expressions import LiteFrameExprEvaluator


def _inner_expr(result, col_index=0):
    """Return the inner expression for a column in a projection result."""
    return result.data.op.projections[col_index].expr


def _eval_projection(result, df):
    """Evaluate a projection result's expressions against *df* via the SPE
    evaluator and return a DataFrame."""
    evaluator = LiteFrameExprEvaluator(df)
    out = {}
    for proj in result.data.op.projections:
        out[proj.name] = evaluator.eval(proj.expr)
    return pd.DataFrame(out)
