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

import numpy as np
import pytest

from maxframe import tensor as mt
from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.learn.linear_model import LogisticRegressionFitAdapter
from maxframe.learn.linear_model._logistic import LogisticRegressionFitOp


def test_logistic_fit_codegen_binary():
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((20, 3))
    y_np = rng.integers(0, 2, 20)

    op = LogisticRegressionFitOp(
        X=X_np,
        y=y_np,
        fit_intercept=True,
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=100,
        tol=1e-4,
        n_classes=2,
    )

    # Call op with maxframe tileables to populate op.inputs and op.outputs.
    # Keep the returned tileables alive: op.outputs only holds weakrefs, and
    # without an external strong reference the output cycle can be GC'd.
    X_mt = mt.asarray(X_np)
    y_mt = mt.asarray(y_np)
    op_outputs = op(X_mt, y_mt)
    assert len(op_outputs) == 3  # coef, n_iter, warn_msg

    adapter = LogisticRegressionFitAdapter()
    context = SPECodeContext()
    codes = adapter.generate_code(op, context)
    assert len(codes) == 1
    assert "LogisticRegressionFitAdapter._run_fit" in codes[0]


def test_logistic_fit_execution_binary():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression as SkLR

    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((50, 3))
    y_np = rng.integers(0, 2, 50)
    # Guarantee two-class labels to avoid sklearn failures
    if len(np.unique(y_np)) < 2:
        y_np[0] = 0
        y_np[1] = 1

    op = LogisticRegressionFitOp(
        X=X_np,
        y=y_np,
        fit_intercept=True,
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=100,
        tol=1e-4,
        n_classes=2,
    )

    # Call op with maxframe tileables to populate op.inputs and op.outputs.
    # Keep the returned tileables alive: op.outputs only holds weakrefs, and
    # without an external strong reference the output cycle can be GC'd.
    X_mt = mt.asarray(X_np)
    y_mt = mt.asarray(y_np)
    op_outputs = op(X_mt, y_mt)
    assert len(op_outputs) == 3  # coef, n_iter, warn_msg

    adapter = LogisticRegressionFitAdapter()
    context = SPECodeContext()
    codes = adapter.generate_code(op, context)

    # Resolve the input and output variable names from the context
    x_var = context.get_input_tileable_variable(op.inputs[0])
    y_var = context.get_input_tileable_variable(op.inputs[1])
    out_vars = [context.get_output_tileable_variable(out) for out in op_outputs]
    coef_var = out_vars[0]

    # Simulate SPE execution: provide X and y as inputs, plus constants
    ns = {x_var: X_np, y_var: y_np}
    ns.update(context.constants)
    # Make the adapter class available for the static method call
    ns["LogisticRegressionFitAdapter"] = LogisticRegressionFitAdapter
    exec("\n".join(codes), ns)

    # Verify against direct sklearn fit
    sk_model = SkLR(penalty="l2", C=1.0, solver="lbfgs", max_iter=100, tol=1e-4)
    sk_model.fit(X_np, y_np)
    coef = ns[coef_var]
    if op.fit_intercept:
        expected_coef = np.concatenate([sk_model.coef_.ravel(), sk_model.intercept_])
    else:
        expected_coef = sk_model.coef_.ravel()
    np.testing.assert_array_almost_equal(coef, expected_coef, decimal=4)
