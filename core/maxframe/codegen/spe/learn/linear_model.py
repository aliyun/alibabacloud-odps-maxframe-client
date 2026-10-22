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

import warnings
from typing import List

import numpy as np

from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.learn.linear_model._logistic import LogisticRegressionFitOp


@register_op_adapter(LogisticRegressionFitOp)
class LogisticRegressionFitAdapter(SPEOperatorAdapter):
    @staticmethod
    def _run_fit(
        X,
        y,
        sample_weight=None,
        coef=None,
        penalty="l2",
        C=1.0,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=100,
        tol=1e-4,
        l1_ratio=None,
        dual=False,
        intercept_scaling=1.0,
        n_classes=2,
        random_state=None,
        warm_start=False,
    ):
        from sklearn.linear_model import LogisticRegression as SkLR

        # Map MaxFrame penalty to sklearn: None means no penalty
        sk_penalty = None if penalty is None or penalty == "none" else penalty

        model = SkLR(
            penalty=sk_penalty,
            C=C if C is not None else 1.0,
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            dual=dual,
            intercept_scaling=intercept_scaling,
            random_state=random_state,
            warm_start=warm_start,
            l1_ratio=l1_ratio,
        )

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if warm_start and coef is not None:
            # Pack initial coefficients into model
            if n_classes == 2:
                if fit_intercept:
                    model.coef_ = coef[:-1].reshape(1, -1)
                    model.intercept_ = np.array([coef[-1]])
                else:
                    model.coef_ = coef.reshape(1, -1)
                    model.intercept_ = np.array([0.0])
            else:
                if fit_intercept:
                    model.coef_ = coef[:-1].T
                    model.intercept_ = coef[-1]
                else:
                    model.coef_ = coef.T
                    model.intercept_ = np.zeros(n_classes)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X, y, **fit_kwargs)
            warn_msg = str(w[-1].message) if w else None

        n_iter = int(model.n_iter_[0])

        # Pack coef and intercept into expected shape
        if n_classes == 2:
            if fit_intercept:
                result_coef = np.concatenate([model.coef_.ravel(), model.intercept_])
            else:
                result_coef = model.coef_.ravel()
        else:
            if fit_intercept:
                result_coef = np.concatenate(
                    [model.coef_.T, model.intercept_.reshape(1, -1)]
                )
            else:
                result_coef = model.coef_.T

        return result_coef, n_iter, warn_msg

    def generate_code(
        self, op: LogisticRegressionFitOp, context: SPECodeContext
    ) -> List[str]:
        context.register_import(__name__, from_item=type(self).__name__)

        # Outputs: coef, n_iter, warn_msg
        out_vars = [context.get_output_tileable_variable(out) for out in op.outputs]
        out_vars_str = ", ".join(out_vars)

        # Pass op.inputs[0] and op.inputs[1] (tileables) so translate_var
        # resolves them to the correct input variable names.
        args = self._translate_call_args(
            context,
            op.inputs[0],
            op.inputs[1],
            sample_weight=op.sample_weight,
            coef=op.coef,
            penalty=op.penalty,
            C=op.C,
            fit_intercept=op.fit_intercept,
            solver=op.solver,
            max_iter=op.max_iter,
            tol=op.tol,
            l1_ratio=op.l1_ratio,
            dual=op.dual,
            intercept_scaling=op.intercept_scaling,
            n_classes=op.n_classes,
            random_state=op.random_state,
            warm_start=op.warm_start,
        )
        return [f"{out_vars_str} = {type(self).__name__}._run_fit({', '.join(args)})"]
