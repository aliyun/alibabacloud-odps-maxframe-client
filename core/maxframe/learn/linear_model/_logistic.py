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

import numbers
import warnings

import numpy as np

from maxframe import opcodes
from maxframe import tensor as mt
from maxframe.core import OutputType
from maxframe.learn.core import BaseEstimator
from maxframe.learn.linear_model._base import LinearClassifierMixin
from maxframe.learn.linear_model._glm import GLMFitOp
from maxframe.learn.linear_model._utils import (
    _apply_balanced_weight,
    _dict_sample_weight,
)
from maxframe.learn.utils.multiclass import check_classification_targets
from maxframe.learn.utils.validation import check_is_fitted
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    Float64Field,
    Int32Field,
    StringField,
)
from maxframe.session import execute as execute_tileables
from maxframe.tensor.special import softmax


def _check_solver(solver):
    all_solvers = ["lbfgs", "newton-cg", "newton-cholesky", "liblinear"]
    solver = solver.lower()
    if solver not in all_solvers:
        raise ValueError(
            "Logistic Regression supports only solvers in %s, got"
            " %s." % (all_solvers, solver)
        )
    return solver


def _check_solver_penalty(solver, penalty):
    """Validate that the penalty is supported by the solver."""
    _solver_penalty_compat = {
        "lbfgs": ["l1", "l2", "elasticnet", "none"],
        "newton-cg": ["l2", "none"],
        "newton-cholesky": ["l2", "none"],
        "liblinear": ["l1", "l2"],
    }
    # Treat None penalty as "none" for compatibility check
    _penalty = "none" if penalty is None else penalty
    supported = _solver_penalty_compat.get(solver, [])
    if _penalty not in supported:
        raise ValueError(
            "Solver %s supports only penalties %s, got %s."
            % (solver, supported, penalty)
        )


def _check_penalty(penalty):
    all_penalties = ["l1", "l2", "elasticnet", "none"]
    if penalty is not None and penalty not in all_penalties:
        raise ValueError(
            "Logistic Regression supports only penalties in %s, got"
            " %s." % (all_penalties, penalty)
        )
    return penalty


class LogisticRegressionFitOp(GLMFitOp):
    """Fit operand for LogisticRegression.

    Inherits the shared GLM fields (X, y, sample_weight, coef, fit_intercept,
    alpha, l1_ratio, penalty, max_iter, tol, random_state, warm_start) from
    :class:`GLMFitOp`. Adds LogisticRegression-specific fields (``C``,
    ``solver``, ``dual``, ``intercept_scaling``, ``class_weight``,
    ``n_classes``). Overrides ``__call__`` for the binary-vs-multiclass
    coefficient shape.
    """

    _op_type_ = opcodes.LOGISTIC_REGRESSION_FIT

    n_classes = Int32Field("n_classes", None)
    C = Float64Field("C", default=None)
    solver = StringField("solver", default=None)
    dual = BoolField("dual", default=False)
    intercept_scaling = Float64Field("intercept_scaling", default=1.0)
    class_weight = AnyField("class_weight", default=None)

    def __call__(self, X, y, sample_weight=None, coef=None):
        X = mt.asarray(X)
        y = mt.asarray(y)
        sample_weight = mt.asarray(sample_weight) if sample_weight is not None else None
        coef = mt.asarray(coef) if coef is not None else None

        inputs = [X, y]
        if sample_weight is not None:
            inputs.append(sample_weight)
        if coef is not None:
            inputs.append(coef)

        self._output_types = [OutputType.tensor, OutputType.scalar, OutputType.object]

        kws = []
        # build coeff output
        n_features = X.shape[1]
        n_classes = self.n_classes
        if self.fit_intercept:
            shape = (n_features + 1, n_classes)
        else:
            shape = (n_features, n_classes)
        if n_classes == 2:
            # outputs binomial classification result
            shape = shape[:1]
        kws.append({"shape": shape, "dtype": np.dtype(float)})

        # build n_iter output and warn_msg output
        kws.extend([{"shape": (), "dtype": np.dtype(int)}, {"shape": ()}])
        return self.new_tileables(inputs, kws=kws)


class LogisticRegression(LinearClassifierMixin, BaseEstimator):
    """
    Logistic Regression (aka logit, MaxEnt) classifier.

    This class implements regularized logistic regression using the
    specified solver. **Note that regularization is applied by default**.
    It can handle both dense and sparse input. Use C-ordered arrays or
    CSR matrices containing 64-bit floats for optimal performance; any
    other input format will be converted (and copied).

    .. note::
       This is a MaxFrame distributed implementation. Supported solvers are
       ``lbfgs``, ``newton-cg``, ``newton-cholesky``, and ``liblinear``.
       The ``multi_class`` parameter is intentionally not provided, following
       scikit-learn's deprecation (removed in sklearn 1.8). Multiclass
       problems always use the multinomial loss. The ``liblinear`` solver
       supports binary classification only; use
       ``OneVsRestClassifier`` for OvR multiclass.

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        Used to specify the norm used in the penalization. Not all penalties
        are supported by all solvers:

        ==========  ================================
        Penalty     Supported solvers
        ==========  ================================
        l1          liblinear
        l2          lbfgs, newton-cg, newton-cholesky, liblinear
        elasticnet  lbfgs
        none        lbfgs, newton-cg, newton-cholesky
        ==========  ================================

        If 'none', no regularization is applied.

    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Smaller values specify stronger regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    random_state : int, RandomState instance, default=None
        Seed for the random number generator used to initialize coefficients.

    solver : {'lbfgs', 'newton-cg', 'newton-cholesky', 'liblinear'}, default='lbfgs'
        Algorithm to use in the optimization problem.

        - ``lbfgs``: Uses scipy.optimize.minimize with L-BFGS-B.
          Supports l1, l2, elasticnet, and none penalties.
          Multiclass: multinomial.
        - ``newton-cg``: Newton's method with CG linear solver.
          Supports l2 and none penalties only.
          Multiclass: multinomial.
        - ``newton-cholesky``: Newton's method with Cholesky factorization.
          Supports l2 and none penalties only.
          Multiclass: multinomial.
        - ``liblinear``: Coordinate descent (wraps LIBLINEAR).
          Supports l1 and l2 penalties only.
          Binary classification only (raises ValueError for n_classes > 2).

        .. note::
           ``newton-cg`` and ``newton-cholesky`` compute the full Hessian
           matrix and require it to fit into AM memory. For high-dimensional
           data, ``lbfgs`` is recommended.

    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

        .. note::
           This parameter is only used by the ``liblinear`` solver.
           For other solvers, this parameter has no effect.

    intercept_scaling : float, default=1.0
        Useful only when solver='liblinear' and fit_intercept=True. In this
        case, the intercept term is scaled by intercept_scaling (i.e. a
        "synthetic" feature with constant value equal to intercept_scaling
        is added to the instance vector). The intercept becomes
        intercept_scaling * synthetic_feature_weight.

        .. note::
           The synthetic feature weight is subject to l1/l2 regularization
           as all other features. To lessen the effect of regularization on
           the intercept term, increase intercept_scaling.

        .. note::
           This parameter is only used by the ``liblinear`` solver.
           For other solvers, this parameter has no effect.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        .. note::
           class_weight is converted to per-sample weights before fitting and
           combined with sample_weight if both are provided. This is equivalent
           to passing class_weight directly for all solvers including liblinear.

    max_iter : int, default=300
        Maximum number of iterations taken for the solver to converge.

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for
        verbosity.

        .. note:: Not yet implemented. Currently has no effect on the
           optimization process.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        as initialization, otherwise, just erase the previous solution.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``.
        Only used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is
        equivalent to using ``penalty='l2'``, while ``l1_ratio=1`` is
        equivalent to using ``penalty='l1'``.

    Attributes
    ----------

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.

    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape (1,) when the given problem is binary.

    n_iter_ : int
        Actual number of iterations for all classes.

    classes_ : ndarray of shape (n_classes,)
        A list of class labels known to the classifier.

    See Also
    --------
    SGDClassifier : Incrementally trained logistic regression (when given
        the parameter ``loss="log"``).
    LogisticRegressionCV : Logistic regression with built-in cross validation.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from maxframe.learn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(random_state=0).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    """

    def __init__(
        self,
        penalty="l2",
        *,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        random_state=None,
        solver="lbfgs",
        max_iter=300,
        verbose=0,
        warm_start=False,
        l1_ratio=None,
        dual=False,
        intercept_scaling=1.0,
        class_weight=None,
    ):
        self.penalty = penalty
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.C = C
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.warm_start = warm_start
        self.dual = dual
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight

    def fit(
        self, X, y, sample_weight=None, execute=False, session=None, run_kwargs=None
    ):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        Regardless of the ``execute`` parameter, this method always performs
        a small synchronous execution to determine the number of classes
        (``n_classes``) from the label data. This is required to decide the
        output dimensions of the model coefficients.
        """
        solver = _check_solver(self.solver)
        _check_penalty(self.penalty)
        _check_solver_penalty(solver, self.penalty)

        # Validate dual parameter
        if self.dual:
            if solver != "liblinear":
                warnings.warn(
                    "Parameter dual is only used by liblinear solver. "
                    f"Got solver={solver}. Parameter will be ignored."
                )
            elif self.penalty != "l2":
                raise ValueError(
                    "Parameter dual=True is only supported for l2 penalty "
                    "with liblinear solver. Got penalty=%s." % self.penalty
                )

        # Validate intercept_scaling parameter
        if self.intercept_scaling != 1.0 and solver != "liblinear":
            warnings.warn(
                "Parameter intercept_scaling is only used by liblinear solver. "
                f"Got solver={solver}. Parameter will be ignored."
            )

        # Validate class_weight parameter
        if self.class_weight is not None:
            if not (
                self.class_weight == "balanced" or isinstance(self.class_weight, dict)
            ):
                raise ValueError(
                    "class_weight must be 'balanced', a dict, or None. "
                    f"Got {type(self.class_weight).__name__}: {self.class_weight}"
                )

        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)

        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iteration must be positive;"
                " got (max_iter=%r)" % self.max_iter
            )

        l1_ratio = self.l1_ratio
        if self.penalty == "elasticnet" and l1_ratio is None:
            raise ValueError("l1_ratio must be specified when penalty is elasticnet.")

        if self.penalty == "l1":
            l1_ratio = 1.0
        elif self.penalty == "l2":
            l1_ratio = 0.0
        elif self.penalty == "elasticnet":
            pass  # l1_ratio already set from self.l1_ratio
        elif self.penalty == "none" or self.penalty is None:
            l1_ratio = 0.0
        else:
            raise ValueError(
                "Unsupported penalty type: %s. Supported penalties: "
                "'l1', 'l2', 'elasticnet', 'none'." % self.penalty
            )

        if self.l1_ratio is not None and l1_ratio != self.l1_ratio:
            warnings.warn(
                "Parameter l1_ratio is ignored when penalty is specified as l1 or l2."
            )

        _dtype = [mt.float64, mt.float32]

        X, y = self._validate_data(X, y, accept_sparse="csr", dtype=_dtype, order="C")

        _, y = check_classification_targets(y, return_value=True)

        self.classes_ = mt.unique(y)
        n_classes = self.classes_.execute().shape[0]

        if n_classes < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes in the data, "
                "but only %d class was found." % n_classes
            )

        if solver == "liblinear" and n_classes > 2:
            raise ValueError(
                "liblinear solver supports only binary classification. "
                "For multiclass, use another solver (lbfgs, newton-cg, "
                "newton-cholesky) or wrap the estimator in "
                "OneVsRestClassifier."
            )

        # Convert class_weight to per-sample weights.
        # For dict class_weight, this must happen while y still holds the original
        # class labels so that dict keys like {"cat": 2.0, "dog": 1.0} can be
        # correctly matched. After conversion we set class_weight=None and let the
        # tiler work only with (effective) sample_weight — which is equivalent for
        # all solvers including liblinear (verified against sklearn's _fit_liblinear).
        class_weight = self.class_weight
        if isinstance(class_weight, dict):
            # dict-based weights: look up each sample's label in the dict.
            # Use apply_chunk on original y (before encoding) so that dict keys
            # match the user's original class labels.
            cw_sample_weight = y.mf.apply_chunk(
                _dict_sample_weight,
                weight_dict=class_weight,
                dtype=np.float64,
                shape=y.shape,
            )

            # Combine with existing sample_weight
            if sample_weight is not None:
                sample_weight = cw_sample_weight * mt.asarray(sample_weight)
            else:
                sample_weight = cw_sample_weight

            # class_weight is now resolved into sample_weight
            class_weight = None

        # Encode y to {0, 1, ..., n_classes-1} so that the loss function
        # computes correct gradients (e.g. HalfBinomialLoss expects y ∈ {0,1}).
        # self.classes_ preserves the original labels for prediction.
        y = mt.searchsorted(self.classes_, y)

        # Handle balanced class_weight after encoding, since it only needs
        # class counts which are the same whether y is original or encoded.
        if class_weight == "balanced":
            # balanced weight for class k: n_samples / (n_classes * count_k)
            class_counts = mt.bincount(y, minlength=n_classes)
            n_samples_val = mt.sum(class_counts)
            # balanced_weights[k] = n_samples / (n_classes * count_k)
            balanced_weights = n_samples_val / (n_classes * class_counts)

            cw_sample_weight = y.mf.apply_chunk(
                _apply_balanced_weight,
                args=(balanced_weights,),
                dtype=np.float64,
                shape=y.shape,
            )

            # Combine with existing sample_weight
            if sample_weight is not None:
                sample_weight = cw_sample_weight * mt.asarray(sample_weight)
            else:
                sample_weight = cw_sample_weight

            # class_weight is now resolved into sample_weight
            class_weight = None

        if not self.warm_start or not hasattr(self, "coef_"):
            coef = None
        elif self.fit_intercept:
            # coef_ is (1, n_features) or (n_classes, n_features)
            # intercept_ is (1,) or (n_classes,)
            # Tiler expects: binary (n_features+1,) or multiclass (n_features+1, n_classes)
            if n_classes == 2:
                # Concatenate (1, n_features) and (1,) flattened -> (n_features+1,)
                coef = mt.concatenate(
                    [self.coef_.reshape((-1,)), self.intercept_.reshape((-1,))]
                )
            else:
                # coef_ is (n_classes, n_features), intercept_ is (n_classes,)
                # Need (n_features+1, n_classes): append intercept as last row
                coef = mt.concatenate(
                    [self.coef_.T, self.intercept_.reshape((1, -1))], axis=0
                )
        else:
            if n_classes == 2:
                # coef_ is (1, n_features), tiler expects (n_features,)
                coef = self.coef_.reshape((-1,))
            else:
                # coef_ is (n_classes, n_features), tiler expects (n_features, n_classes)
                coef = self.coef_.T

        op = LogisticRegressionFitOp(
            X=X,
            y=y,
            sample_weight=sample_weight,
            coef=coef,
            n_classes=n_classes,
            penalty=self.penalty,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
            solver=solver,
            max_iter=self.max_iter,
            l1_ratio=l1_ratio,
            dual=self.dual,
            intercept_scaling=self.intercept_scaling,
            class_weight=class_weight,
        )
        coef, n_iter, warn_msg = op(X, y, sample_weight=sample_weight, coef=coef)

        # ========== Post processing =======
        if self.fit_intercept:
            if n_classes == 2:
                # Binary: coef is 1D (n_features + 1,)
                # intercept_ is last element, coef_ is (1, n_features)
                self.intercept_ = coef[-1].reshape((1,))
                self.coef_ = coef[:-1].reshape((1, -1))
            else:
                # Multiclass: coef is 2D (n_features + 1, n_classes)
                # intercept_ is last row transposed -> (n_classes,)
                # coef_ is remaining rows transposed -> (n_classes, n_features)
                self.intercept_ = coef[-1, :].reshape((n_classes,))
                self.coef_ = coef[:-1, :].T
        else:
            if n_classes == 2:
                # Binary without intercept: coef is 1D (n_features,)
                # reshape to (1, n_features) to match sklearn convention
                self.intercept_ = mt.zeros(1)
                self.coef_ = coef.reshape((1, -1))
            else:
                # Multiclass without intercept: coef is 2D (n_features, n_classes)
                # transpose to (n_classes, n_features) to match sklearn convention
                self.intercept_ = mt.zeros(n_classes)
                self.coef_ = coef.T
        self.n_iter_ = n_iter

        if execute:
            execute_tileables(
                self.coef_,
                self.intercept_,
                n_iter,
                warn_msg,
                session=session,
                **(run_kwargs or {}),
            )
            warn_msg_val = warn_msg.fetch(session=session)
            if warn_msg_val is not None:
                try:
                    from sklearn.exceptions import ConvergenceWarning

                    warnings.warn(warn_msg_val, ConvergenceWarning)
                except ImportError:
                    warnings.warn(warn_msg_val)
        return self

    def predict_proba(self, X, execute=False, session=None, run_kwargs=None):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        For binary classification, the sigmoid function is used to find
        the predicted probability of each class. For multiclass, the
        softmax function is used.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)
        decision = self.decision_function(X)

        if decision.ndim == 1:
            # Binary case: convert decision scores to 2-class probabilities
            # using sigmoid function
            proba_class_1 = 1.0 / (1.0 + mt.exp(-decision))
            proba = mt.stack([1.0 - proba_class_1, proba_class_1], axis=1)
        else:
            proba = softmax(decision, axis=1)
        if execute:
            proba.execute(session=session, **(run_kwargs or {}))
        return proba

    def predict_log_proba(self, X, execute=False, session=None, run_kwargs=None):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        res = mt.log(self.predict_proba(X))
        if execute:
            res.execute(session=session, **(run_kwargs or {}))
        return res
