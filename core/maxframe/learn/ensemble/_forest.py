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

"""Distributed random forest classifiers and regressors.

This module provides sklearn-compatible estimators for random forest
learning on MaxFrame. All trees in the forest are trained simultaneously
using a single pass over the data per tree level, which is significantly
more efficient than training trees independently.
"""

from typing import List

import numpy as np

from maxframe import opcodes
from maxframe import tensor as mt
from maxframe.core import ENTITY_TYPE, EntityData, OutputType
from maxframe.core.operator import Operator
from maxframe.learn.core import (
    BaseEstimator,
    ClassifierMixin,
    LearnOperatorMixin,
    RegressorMixin,
)
from maxframe.learn.utils.validation import check_is_fitted
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    Float64Field,
    Int32Field,
    KeyField,
    StringField,
)
from maxframe.session import execute as execute_tileables
from maxframe.tensor.random import RandomStateField


class RandomForestFitOp(LearnOperatorMixin, Operator):
    """Operator for fitting a distributed random forest."""

    _op_type_ = opcodes.RANDOM_FOREST_FIT

    X = KeyField("X")
    y = KeyField("y")
    sample_weight = KeyField("sample_weight", default=None)
    n_estimators = Int32Field("n_estimators", default=100)
    criterion = StringField("criterion", default=None)
    max_depth = Int32Field("max_depth", default=None)
    max_bins = Int32Field("max_bins", default=32)
    min_samples_split = AnyField("min_samples_split", default=2)
    min_samples_leaf = AnyField("min_samples_leaf", default=1)
    min_weight_fraction_leaf = Float64Field("min_weight_fraction_leaf", default=0.0)
    max_features = AnyField("max_features", default=None)
    bootstrap = BoolField("bootstrap", default=True)
    oob_score = BoolField("oob_score", default=False)
    random_state = RandomStateField("random_state", default=None)
    min_impurity_decrease = Float64Field("min_impurity_decrease", default=0.0)
    max_memory_in_mb = Int32Field("max_memory_in_mb", default=256)
    max_samples = AnyField("max_samples", default=None)
    feature_subset_strategy = StringField("feature_subset_strategy", default="auto")
    is_classification = BoolField("is_classification", default=True)
    n_classes = Int32Field("n_classes", default=2)
    n_features = Int32Field("n_features", default=1)

    @property
    def output_limit(self) -> int:
        return 5

    @classmethod
    def _set_inputs(cls, op: "RandomForestFitOp", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        input_iter = iter(op._inputs)
        op.X = next(input_iter)
        op.y = next(input_iter)
        if isinstance(op.sample_weight, ENTITY_TYPE):
            op.sample_weight = next(input_iter)

    def __call__(self, X, y, sample_weight=None):
        self.X = X
        self.y = y
        self.sample_weight = sample_weight

        inputs = [X, y]
        if isinstance(sample_weight, ENTITY_TYPE):
            inputs.append(sample_weight)

        self._output_types = [
            OutputType.object,  # forest model (list of tree states)
            OutputType.tensor,  # labels
            OutputType.tensor,  # feature_importances
            OutputType.object,  # oob_score (if computed)
            OutputType.object,  # metadata
        ]
        kws = [
            {"shape": (), "dtype": np.dtype(object)},  # forest model
            {"dtype": np.dtype(int), "shape": (X.shape[0],)},  # labels
            {"dtype": np.float64, "shape": (X.shape[1],)},  # feature_importances
            {"shape": (), "dtype": np.dtype(object)},  # oob_score
            {"shape": (), "dtype": np.dtype(object)},  # metadata
        ]
        return self.new_tileables(inputs, kws=kws)


class RandomForestPredictOp(LearnOperatorMixin, Operator):
    """Operator for predicting with a fitted random forest.

    The fitted ``forest_state`` is carried as a tileable input (KeyField),
    not fetched eagerly.  The tiler materializes it during tiling so that
    ``predict(execute=False)`` builds a composable graph without executing.
    """

    _op_type_ = opcodes.RANDOM_FOREST_PREDICT

    X = KeyField("X")
    forest_state = KeyField("forest_state", default=None)
    n_classes = Int32Field("n_classes", default=2)
    is_classification = BoolField("is_classification", default=True)
    n_estimators = Int32Field("n_estimators", default=100)

    @property
    def output_limit(self) -> int:
        return 2

    @classmethod
    def _set_inputs(cls, op: "RandomForestPredictOp", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.X = op._inputs[0]
        if op.forest_state is not None and len(op._inputs) > 1:
            op.forest_state = op._inputs[1]

    def __call__(self, X):
        self.X = X
        # forest_state is a tileable input; the tiler fetches it during
        # tiling so graph construction stays lazy.
        inputs = [X]
        if self.forest_state is not None:
            inputs.append(self.forest_state)

        if self.is_classification:
            self._output_types = [OutputType.tensor, OutputType.tensor]
            kws = [
                {"dtype": np.dtype(int), "shape": (X.shape[0],)},  # labels
                {"dtype": np.float64, "shape": (X.shape[0], self.n_classes)},  # proba
            ]
        else:
            self._output_types = [OutputType.tensor, OutputType.tensor]
            kws = [
                {"dtype": np.float64, "shape": (X.shape[0],)},  # predictions
                {"dtype": np.float64, "shape": (X.shape[0],)},  # same
            ]
        return self.new_tileables(inputs, kws=kws)


def _forest_predict_raw(
    estimator, X, is_classification, n_classes, execute, session, run_kwargs
):
    """Run the forest prediction op and return (labels, proba_or_predictions).

    Shared by RandomForestClassifier.predict / predict_proba and
    RandomForestRegressor.predict.
    """
    check_is_fitted(estimator)
    X = mt.tensor(X, dtype=np.float64)
    op = RandomForestPredictOp(
        forest_state=estimator.forest_model_,
        n_classes=n_classes if is_classification else 1,
        is_classification=is_classification,
        n_estimators=estimator.n_estimators,
    )
    labels, secondary = op(X)
    if execute:
        secondary.execute(session=session, **(run_kwargs or {}))
    return labels, secondary


class RandomForestClassifier(ClassifierMixin, BaseEstimator):
    """A distributed random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.

    This distributed implementation trains all trees simultaneously using
    a single pass over the data per tree level, which is significantly more
    efficient than training trees independently. Bootstrap sampling is
    handled via per-tree Poisson-distributed sample weights at the
    partition level.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria
        are "gini" for the Gini impurity and "entropy" for the information
        gain.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded
        until all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    max_bins : int, default=32
        The maximum number of bins used to discretize continuous features.
        See :class:`~maxframe.learn.tree.DecisionTreeClassifier` for
        details.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights required
        to be at a leaf node.

    max_features : {"sqrt", "log2"}, int or float, default="sqrt"
        The number of features to consider when looking for the best split.
        Note: "auto" is not supported; use "sqrt" explicitly.

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features))` features are considered.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False,
        the whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization
        score.

        .. warning::
            Out-of-bag scoring is not yet implemented. Setting
            ``oob_score=True`` will raise ``NotImplementedError``.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples
        used when building trees (if ``bootstrap=True``) and the sampling
        of the features to consider when looking for the best split at
        each node.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max(round(n_samples * max_samples), 1)`
          samples. Thus, `max_samples` should be in the interval
          `(0.0, 1.0]`.

    max_memory_in_mb : int, default=256
        The maximum memory in MB to use for aggregating statistics for a
        group of nodes during training. When training many trees, memory
        is shared across all trees at the same level. Fewer trees per
        group are processed when the budget is exceeded.

    feature_subset_strategy : {"auto", "sqrt", "log2", "onethird", "all"} \
            or int, default="auto"
        The strategy for selecting features at each node split:

        - "auto": "sqrt" for classification, "onethird" for regression.
        - "sqrt": ceil(sqrt(n_features)) features.
        - "log2": max(1, ceil(log2(n_features))) features.
        - "onethird": ceil(n_features / 3) features.
        - "all": all features (equivalent to bagging without feature
          subsampling).
        - If int: the exact number of features.

    Attributes
    ----------
    estimator_ : DecisionTreeClassifier
        The child estimator template used to create the collection of
        fitted sub-estimators.

    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The class labels.

    n_classes_ : int or list
        The number of classes.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.

    n_outputs_ : int
        The number of outputs when fit is performed.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances. Currently returns all zeros
        (not yet implemented). Tracked as a known limitation.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag
        estimate. This attribute exists only when ``oob_score`` is True.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the
        training set. This attribute exists only when ``oob_score``
        is True.

    See Also
    --------
    DecisionTreeClassifier : A distributed decision tree classifier.
    RandomForestRegressor : A distributed random forest regressor.

    Notes
    -----
    **Distributed Algorithm**

    All trees in the forest are trained simultaneously using a single
    distributed data pass per tree level. This is much more efficient
    than training trees independently, as the cost of a data pass is
    amortized across all trees.

    The training process:

    1. **Bootstrap sampling**: For each tree, per-sample weights are drawn
       from a Poisson(1) distribution at each partition, effectively
       creating bootstrap samples without data movement.
    2. **Feature subsampling**: At each node, a random subset of features
       (controlled by ``feature_subset_strategy``) is selected. The same
       random seed ensures consistent feature selection across partitions.
    3. **Sufficient statistics aggregation**: Per-bin class counts are
       aggregated across all partitions and reduced via a shuffle
       operation for all trees simultaneously.
    4. **Split evaluation**: Best splits are found independently for each
       tree using the merged statistics.

    **Limitations compared to scikit-learn**

    In addition to the limitations documented in
    :class:`~maxframe.learn.tree.DecisionTreeClassifier`:

    - ``n_jobs``: Not applicable. Distributed parallelism is handled
      automatically by the MaxCompute execution engine.
    - ``warm_start``: Not supported. Incremental forest training is not
      available in the distributed setting.
    - ``class_weight``: Not yet supported.
    - ``oob_score``: Not yet implemented.
    - ``ccp_alpha``: Cost-complexity pruning is not supported.
    - ``monotonic_cst``: Monotonicity constraints are not supported.

    **Performance Considerations**

    - Training time scales with the deepest tree, not the average depth,
      since all trees are grown level-by-level together.
    - The total memory per iteration is proportional to
      ``n_estimators * n_features * max_bins * n_classes``. For very
      large forests, reduce ``max_bins`` or ``max_memory_in_mb``.
    - The number of distributed communication rounds equals the maximum
      tree depth across all trees.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
           2001.

    Examples
    --------
    >>> from maxframe.learn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier(n_estimators=100, max_depth=5)
    >>> clf.fit(X_train, y_train)
    RandomForestClassifier(max_depth=5, n_estimators=100)
    >>> predictions = clf.predict(X_test)
    >>> proba = clf.predict_proba(X_test)
    """

    def __init__(
        self,
        *,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        max_bins=32,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        bootstrap=True,
        oob_score=False,
        random_state=None,
        verbose=0,
        max_samples=None,
        max_memory_in_mb=256,
        feature_subset_strategy="auto",
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose
        self.max_samples = max_samples
        self.max_memory_in_mb = max_memory_in_mb
        self.feature_subset_strategy = feature_subset_strategy

    def fit(
        self, X, y, sample_weight=None, execute=False, session=None, run_kwargs=None
    ):
        """Fit a distributed random forest classifier.

        Parameters
        ----------
        X : {array-like, tensor} of shape (n_samples, n_features)
            The training input samples.
        y : {array-like, tensor} of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
        execute : bool, default=False
            Whether to execute the computation graph immediately.
        session : Session, default=None
            The session to use for execution.
        run_kwargs : dict, default=None
            Additional keyword arguments passed to ``execute``.

        Returns
        -------
        self : RandomForestClassifier
            Fitted estimator.

        Notes
        -----
        Regardless of the ``execute`` parameter, this method always performs
        a small synchronous execution to discover class labels and determine
        ``n_classes_``.
        """
        self._validate_params()
        X = mt.tensor(X, dtype=np.float64)
        n_features = X.shape[1] if X.ndim == 2 else 1
        if isinstance(n_features, float) and np.isnan(n_features):
            raise ValueError(
                "Cannot determine the number of features from X.shape[1] "
                "(it is unknown). Resolve the feature dimension before "
                "calling fit, e.g. X.execute(session=session)."
            )
        # Discover class labels via distributed unique and encode to
        # 0-based integers for classification.  Using mt.unique /
        # mt.searchsorted keeps the operation in MaxFrame instead of
        # pulling the full target array back to the driver.
        y = mt.asarray(y)
        self.classes_ = mt.unique(y)
        self.n_classes_ = self.classes_.execute(
            session=session, **(run_kwargs or {})
        ).shape[0]
        y = mt.searchsorted(self.classes_, y).astype(np.int32)
        if sample_weight is not None:
            sample_weight = mt.tensor(sample_weight, dtype=np.float64)
        self.n_features_ = n_features
        op = RandomForestFitOp(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            max_bins=self.max_bins,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.random_state,
            min_impurity_decrease=0.0,
            max_memory_in_mb=self.max_memory_in_mb,
            max_samples=self.max_samples,
            feature_subset_strategy=self.feature_subset_strategy,
            is_classification=True,
            n_classes=self.n_classes_,
            n_features=self.n_features_,
        )
        forest_model, labels, feature_importances, oob_result, metadata = op(
            X, y, sample_weight
        )
        self.forest_model_ = forest_model
        self.labels_ = labels
        self.feature_importances_ = feature_importances
        self._oob_result = oob_result
        self._metadata = metadata
        if execute:
            execute_tileables(
                self.forest_model_,
                self.labels_,
                self.feature_importances_,
                self._oob_result,
                self._metadata,
                session=session,
                **(run_kwargs or {}),
            )
        return self

    def predict(self, X, execute=False, session=None, run_kwargs=None):
        """Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates.

        Parameters
        ----------
        X : {array-like, tensor} of shape (n_samples, n_features)
            The input samples.
        execute : bool, default=False
            Whether to execute the computation graph immediately.
        session : Session, default=None
            The session to use for execution.
        run_kwargs : dict, default=None
            Additional keyword arguments passed to ``execute``.

        Returns
        -------
        y : tensor of shape (n_samples,)
            The predicted classes.
        """
        labels, _ = _forest_predict_raw(
            self, X, True, self.n_classes_, execute, session, run_kwargs
        )
        # Map 0-based indices back to original class labels
        classes = mt.tensor(self.classes_)
        res = classes[labels]
        if execute:
            res.execute(session=session, **(run_kwargs or {}))
        return res

    def predict_proba(self, X, execute=False, session=None, run_kwargs=None):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed
        as the mean predicted class probabilities of the trees in the
        forest.

        Parameters
        ----------
        X : {array-like, tensor} of shape (n_samples, n_features)
            The input samples.
        execute : bool, default=False
            Whether to execute the computation graph immediately.
        session : Session, default=None
            The session to use for execution.
        run_kwargs : dict, default=None
            Additional keyword arguments passed to ``execute``.

        Returns
        -------
        p : tensor of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        _labels, proba = _forest_predict_raw(
            self, X, True, self.n_classes_, execute, session, run_kwargs
        )
        return proba

    def predict_log_proba(self, X, execute=False, session=None, run_kwargs=None):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : {array-like, tensor} of shape (n_samples, n_features)
            The input samples.
        execute : bool, default=False
            Whether to execute the computation graph immediately.
        session : Session, default=None
            The session to use for execution.
        run_kwargs : dict, default=None
            Additional keyword arguments passed to ``execute``.

        Returns
        -------
        p : tensor of shape (n_samples, n_classes)
            The class log-probabilities of the input samples.
        """
        proba = self.predict_proba(X, session=session, run_kwargs=run_kwargs)
        res = mt.log(proba)
        if execute:
            res.execute(session=session, **(run_kwargs or {}))
        return res

    def _validate_params(self):
        """Validate estimator parameters."""
        if self.criterion not in ("gini", "entropy"):
            raise ValueError(
                f"Criterion '{self.criterion}' is not supported. "
                f"Valid criteria are 'gini' and 'entropy'."
            )
        if self.n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be greater than zero. " f"Got {self.n_estimators}."
            )
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError(
                f"max_depth must be greater than zero or None. "
                f"Got {self.max_depth}."
            )
        if self.max_bins < 2:
            raise ValueError(f"max_bins must be at least 2. Got {self.max_bins}.")
        if self.oob_score:
            raise NotImplementedError(
                "Out-of-bag scoring is not yet implemented in the "
                "distributed random forest."
            )


class RandomForestRegressor(RegressorMixin, BaseEstimator):
    """A distributed random forest regressor.

    A random forest is a meta estimator that fits a number of decision tree
    regressors on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.

    This distributed implementation trains all trees simultaneously using
    a single pass over the data per tree level, which is significantly more
    efficient than training trees independently.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"squared_error"}, default="squared_error"
        The function to measure the quality of a split. Only "squared_error"
        is currently supported in the distributed implementation.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded
        until all leaves contain less than min_samples_split samples.

    max_bins : int, default=32
        The maximum number of bins used to discretize continuous features.
        See :class:`~maxframe.learn.tree.DecisionTreeClassifier` for
        details.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights required
        to be at a leaf node.

    max_features : {"sqrt", "log2"}, int or float, default=1.0
        The number of features to consider when looking for the best split.

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features))` features are considered.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If 1.0, then `max_features=n_features`.

        .. note::
            The default of 1.0 is equivalent to bagged trees and more
            randomness can be achieved by setting smaller values, e.g. 0.3.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False,
        the whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization
        score.

        .. warning::
            Out-of-bag scoring is not yet implemented. Setting
            ``oob_score=True`` will raise ``NotImplementedError``.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples
        and the sampling of the features to consider at each node.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

    max_memory_in_mb : int, default=256
        The maximum memory in MB for aggregating statistics per node group.

    feature_subset_strategy : {"auto", "sqrt", "log2", "onethird", "all"} \
            or int, default="auto"
        The strategy for selecting features at each node split. Default
        "auto" resolves to "onethird" for regression.

    Attributes
    ----------
    estimator_ : DecisionTreeRegressor
        The child estimator template.

    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances. Currently returns all zeros
        (not yet implemented). Tracked as a known limitation.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.

    n_outputs_ : int
        The number of outputs when fit is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag
        estimate. Only when ``oob_score`` is True.

    oob_prediction_ : ndarray of shape (n_samples,)
        Prediction computed with out-of-bag estimate on the training set.
        Only when ``oob_score`` is True.

    See Also
    --------
    DecisionTreeRegressor : A distributed decision tree regressor.
    RandomForestClassifier : A distributed random forest classifier.

    Notes
    -----
    **Limitations compared to scikit-learn**

    In addition to the limitations documented in
    :class:`~maxframe.learn.tree.DecisionTreeClassifier` and
    :class:`~maxframe.learn.ensemble.RandomForestClassifier`:

    - Only "squared_error" criterion is supported. "absolute_error",
      "friedman_mse", and "poisson" are not yet available.
    - The default ``max_features`` is 1.0 (all features), which differs
      from sklearn's original recommendation of n_features/3 [1]_ but
      aligns with more recent empirical findings [2]_.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
           2001.

    .. [2] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
           trees", Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from maxframe.learn.ensemble import RandomForestRegressor
    >>> reg = RandomForestRegressor(n_estimators=100, max_depth=5)
    >>> reg.fit(X_train, y_train)
    RandomForestRegressor(max_depth=5, n_estimators=100)
    >>> predictions = reg.predict(X_test)
    """

    def __init__(
        self,
        *,
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        max_bins=32,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        bootstrap=True,
        oob_score=False,
        random_state=None,
        verbose=0,
        max_samples=None,
        max_memory_in_mb=256,
        feature_subset_strategy="auto",
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose
        self.max_samples = max_samples
        self.max_memory_in_mb = max_memory_in_mb
        self.feature_subset_strategy = feature_subset_strategy

    def fit(
        self, X, y, sample_weight=None, execute=False, session=None, run_kwargs=None
    ):
        """Fit a distributed random forest regressor.

        Parameters
        ----------
        X : {array-like, tensor} of shape (n_samples, n_features)
            The training input samples.
        y : {array-like, tensor} of shape (n_samples,) or (n_samples, n_outputs)
            The target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
        execute : bool, default=False
            Whether to execute the computation graph immediately.
        session : Session, default=None
            The session to use for execution.
        run_kwargs : dict, default=None
            Additional keyword arguments passed to ``execute``.

        Returns
        -------
        self : RandomForestRegressor
            Fitted estimator.
        """
        self._validate_params()
        X = mt.tensor(X, dtype=np.float64)
        n_features = X.shape[1] if X.ndim == 2 else 1
        if isinstance(n_features, float) and np.isnan(n_features):
            raise ValueError(
                "Cannot determine the number of features from X.shape[1] "
                "(it is unknown). Resolve the feature dimension before "
                "calling fit, e.g. X.execute(session=session)."
            )
        y = mt.tensor(y, dtype=np.float64)
        if sample_weight is not None:
            sample_weight = mt.tensor(sample_weight, dtype=np.float64)
        self.n_features_ = n_features
        op = RandomForestFitOp(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            max_bins=self.max_bins,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.random_state,
            min_impurity_decrease=0.0,
            max_memory_in_mb=self.max_memory_in_mb,
            max_samples=self.max_samples,
            feature_subset_strategy=self.feature_subset_strategy,
            is_classification=False,
            n_classes=1,
            n_features=self.n_features_,
        )
        forest_model, labels, feature_importances, oob_result, metadata = op(
            X, y, sample_weight
        )
        self.forest_model_ = forest_model
        self.labels_ = labels
        self.feature_importances_ = feature_importances
        self._oob_result = oob_result
        self._metadata = metadata
        if execute:
            execute_tileables(
                self.forest_model_,
                self.labels_,
                self.feature_importances_,
                self._oob_result,
                self._metadata,
                session=session,
                **(run_kwargs or {}),
            )
        return self

    def predict(self, X, execute=False, session=None, run_kwargs=None):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as
        the mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, tensor} of shape (n_samples, n_features)
            The input samples.
        execute : bool, default=False
            Whether to execute the computation graph immediately.
        session : Session, default=None
            The session to use for execution.
        run_kwargs : dict, default=None
            Additional keyword arguments passed to ``execute``.

        Returns
        -------
        y : tensor of shape (n_samples,)
            The predicted values.
        """
        _labels, predictions = _forest_predict_raw(
            self, X, False, 1, execute, session, run_kwargs
        )
        return predictions

    def _validate_params(self):
        """Validate estimator parameters."""
        if self.criterion != "squared_error":
            raise ValueError(
                f"Criterion '{self.criterion}' is not supported. "
                f"Only 'squared_error' is currently supported in the "
                f"distributed implementation."
            )
        if self.n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be greater than zero. " f"Got {self.n_estimators}."
            )
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError(
                f"max_depth must be greater than zero or None. "
                f"Got {self.max_depth}."
            )
        if self.max_bins < 2:
            raise ValueError(f"max_bins must be at least 2. Got {self.max_bins}.")
        if self.oob_score:
            raise NotImplementedError(
                "Out-of-bag scoring is not yet implemented in the "
                "distributed random forest."
            )
