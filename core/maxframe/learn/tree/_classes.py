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

"""Distributed decision tree classifiers and regressors.

This module provides sklearn-compatible estimators for decision tree
learning on MaxFrame. The implementation uses approximate split finding
with feature binning for distributed execution.
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
from maxframe.learn.tree._tree import Tree
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


class DecisionTreeFitOp(LearnOperatorMixin, Operator):
    """Operator for fitting a distributed decision tree."""

    _op_type_ = opcodes.DECISION_TREE_FIT

    X = KeyField("X")
    y = KeyField("y")
    sample_weight = KeyField("sample_weight", default=None)
    criterion = StringField("criterion", default=None)
    max_depth = Int32Field("max_depth", default=None)
    max_bins = Int32Field("max_bins", default=32)
    min_samples_split = AnyField("min_samples_split", default=2)
    min_samples_leaf = AnyField("min_samples_leaf", default=1)
    min_weight_fraction_leaf = Float64Field("min_weight_fraction_leaf", default=0.0)
    max_features = AnyField("max_features", default=None)
    random_state = RandomStateField("random_state", default=None)
    min_impurity_decrease = Float64Field("min_impurity_decrease", default=0.0)
    max_memory_in_mb = Int32Field("max_memory_in_mb", default=256)
    is_classification = BoolField("is_classification", default=True)
    n_classes = Int32Field("n_classes", default=2)
    n_features = Int32Field("n_features", default=1)

    @property
    def output_limit(self) -> int:
        return 4

    @classmethod
    def _set_inputs(cls, op: "DecisionTreeFitOp", inputs: List[EntityData]):
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
            OutputType.object,
            OutputType.tensor,
            OutputType.tensor,
            OutputType.object,
        ]
        kws = [
            {"shape": (), "dtype": np.dtype(object)},  # tree model
            {"dtype": np.dtype(int), "shape": (X.shape[0],)},  # labels
            {"dtype": np.float64, "shape": (X.shape[1],)},  # feature_importances
            {"shape": (), "dtype": np.dtype(object)},  # metadata
        ]
        return self.new_tileables(inputs, kws=kws)


class DecisionTreePredictOp(LearnOperatorMixin, Operator):
    """Operator for predicting with a fitted decision tree.

    The fitted ``tree_state`` is carried as a tileable input (KeyField),
    not fetched eagerly.  The tiler materializes it during tiling so that
    ``predict(execute=False)`` builds a composable graph without executing.
    """

    _op_type_ = opcodes.DECISION_TREE_PREDICT

    X = KeyField("X")
    tree_state = KeyField("tree_state", default=None)
    n_classes = Int32Field("n_classes", default=2)
    is_classification = BoolField("is_classification", default=True)

    @property
    def output_limit(self) -> int:
        return 2

    @classmethod
    def _set_inputs(cls, op: "DecisionTreePredictOp", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.X = op._inputs[0]
        if op.tree_state is not None and len(op._inputs) > 1:
            op.tree_state = op._inputs[1]

    def __call__(self, X):
        self.X = X
        # tree_state is a tileable input; the tiler fetches it during
        # tiling so graph construction stays lazy.
        inputs = [X]
        if self.tree_state is not None:
            inputs.append(self.tree_state)

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
                {"dtype": np.float64, "shape": (X.shape[0],)},  # same as predictions
            ]
        return self.new_tileables(inputs, kws=kws)


class DecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    """A distributed decision tree classifier.

    This estimator implements a distributed version of the decision tree
    classifier using approximate split finding with feature binning,
    following the approach described in [1]_. The algorithm is designed
    for large-scale datasets that do not fit in a single machine's memory
    and runs on MaxCompute's distributed execution engine.

    Unlike scikit-learn's DecisionTreeClassifier which uses an exact
    greedy algorithm to find optimal split points, this implementation
    discretizes continuous features into bins and finds approximate splits.
    This trades a small amount of split quality for the ability to
    efficiently aggregate statistics across distributed partitions.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria
        are "gini" for the Gini impurity and "entropy" for the information
        gain.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded
        until all leaves are pure or until all leaves contain less than
        min_samples_split samples. Note that very deep trees may require
        many distributed communication rounds.

    max_bins : int, default=32
        The maximum number of bins used to discretize continuous features.
        Higher values improve split approximation quality at the cost of
        increased memory usage and communication overhead. This parameter
        has no equivalent in scikit-learn and is specific to the
        distributed binning algorithm.

        Memory usage per node group is approximately
        ``n_features * max_bins * n_classes * 8`` bytes. When the memory
        budget (controlled by ``max_memory_in_mb``) is exceeded, fewer
        nodes are processed per iteration, increasing the number of data
        passes required.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves
        at least ``min_samples_leaf`` training samples in each of the left
        and right branches.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, {"sqrt", "log2"} or None, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features))` features are considered.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split. When ``max_features < n_features``,
        the algorithm will select ``max_features`` at random at each split
        before finding the best split among them. To obtain a deterministic
        behaviour during fitting, ``random_state`` has to be fixed to an
        integer.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the
        impurity greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number
        of samples at the current node, ``N_t_L`` is the number of samples
        in the left child, and ``N_t_R`` is the number of samples in the
        right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted
        sum, if ``sample_weight`` is passed.

    max_memory_in_mb : int, default=256
        The maximum memory in MB to use for aggregating statistics for a
        group of nodes during training. If the memory budget is exceeded,
        fewer nodes are processed per iteration, increasing the total
        number of data passes. This parameter has no equivalent in
        scikit-learn and is specific to the distributed algorithm.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The class labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing
        the number of classes for each output (multi-output problem).

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances. Currently returns all zeros
        (not yet implemented). Tracked as a known limitation.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values).

    max_features_ : int
        The inferred value of max_features.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit. Defined only when `X` has
        feature names that are all strings.

    n_outputs_ : int
        The number of outputs when fit is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(maxframe.learn.tree._tree.Tree)`` for attributes of the
        Tree object.

    See Also
    --------
    DecisionTreeRegressor : A distributed decision tree regressor.
    RandomForestClassifier : A distributed random forest classifier.

    Notes
    -----
    **Distributed Algorithm**

    This implementation uses a level-wise (breadth-first) training approach
    with approximate split finding, following the algorithm described in [1]_:

    1. Continuous features are discretized into ``max_bins`` bins using
       approximate quantile thresholds computed from a sample of the data.
    2. At each level of the tree, sufficient statistics (per-bin class
       counts for classification, or sum/variance for regression) are
       aggregated across all distributed partitions and then reduced via
       a shuffle operation.
    3. Best splits are found by evaluating impurity improvement at each
       bin boundary using the merged statistics.
    4. Multiple nodes at the same level are processed simultaneously,
       grouped to fit within the ``max_memory_in_mb`` budget.

    **Limitations compared to scikit-learn**

    The following features are **not supported** due to fundamental
    incompatibility with the distributed level-wise training algorithm:

    - ``splitter``: Only "best" splitting is supported. The "random"
      splitter (used by ExtraTrees) is not available because the binning
      approach already provides a form of discretization.
    - ``max_leaf_nodes``: Best-first tree growth is incompatible with
      level-wise distributed training. Use ``max_depth`` to control tree
      size instead.
    - ``ccp_alpha``: Cost-complexity pruning is not currently supported.

    The following features are **deferred to future versions**:

    - ``monotonic_cst``: Monotonicity constraints.
    - ``class_weight``: Class weight balancing.
    - ``oob_score``: Out-of-bag scoring (only applies to forests).

    **Performance Considerations**

    - The number of distributed communication rounds equals the tree depth.
      Deep trees (``max_depth > 15``) may be expensive.
    - Increasing ``max_bins`` improves split quality but increases per-round
      memory and communication. For most datasets, 32-64 bins are sufficient.
    - The ``max_memory_in_mb`` parameter controls how many nodes are processed
      per iteration. Lower values increase the number of data passes but
      prevent out-of-memory errors.

    References
    ----------

    .. [1] L. Breiman, J. Friedman, R. Olshen, and C. Stone,
           "Classification and Regression Trees", Wadsworth, 1984.

    .. [2] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from maxframe.learn.tree import DecisionTreeClassifier
    >>> from maxframe import new_session
    >>> from odps import ODPS
    >>> o = ODPS(...)
    >>> session = new_session(o)
    >>> clf = DecisionTreeClassifier(max_depth=5, max_bins=32)
    >>> clf.fit(X_train, y_train)
    DecisionTreeClassifier(max_bins=32, max_depth=5)
    >>> predictions = clf.predict(X_test)
    >>> predictions.execute()
    >>> predictions.fetch()
    """

    def __init__(
        self,
        *,
        criterion="gini",
        max_depth=None,
        max_bins=32,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        min_impurity_decrease=0.0,
        max_memory_in_mb=256,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.max_memory_in_mb = max_memory_in_mb

    # Explicit data attributes to prevent _get_data_attributes() from
    # scanning all public attributes ending with '_', which would trigger
    # the lazy ``tree_`` property and call ``tree_model_.fetch()`` on
    # unexecuted tileables.
    _data_attributes = ["tree_model_", "labels_", "feature_importances_", "_metadata"]

    def fit(
        self, X, y, sample_weight=None, execute=False, session=None, run_kwargs=None
    ):
        """Fit a distributed decision tree classifier.

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
        self : DecisionTreeClassifier
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
        # Convert sample_weight to tensor if provided
        if sample_weight is not None:
            sample_weight = mt.tensor(sample_weight, dtype=np.float64)
        self.n_features_ = n_features
        op = DecisionTreeFitOp(
            criterion=self.criterion,
            max_depth=self.max_depth,
            max_bins=self.max_bins,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            min_impurity_decrease=self.min_impurity_decrease,
            max_memory_in_mb=self.max_memory_in_mb,
            is_classification=True,
            n_classes=self.n_classes_,
            n_features=self.n_features_,
        )
        tree_model, labels, feature_importances, metadata = op(X, y, sample_weight)
        self.tree_model_ = tree_model
        self.labels_ = labels
        self.feature_importances_ = feature_importances
        self._metadata = metadata
        # Invalidate cached tree representation
        self._tree_obj = None
        if execute:
            execute_tileables(
                self.tree_model_,
                self.labels_,
                self.feature_importances_,
                self._metadata,
                session=session,
                **(run_kwargs or {}),
            )
        return self

    @property
    def tree_(self):
        """The underlying Tree object.

        Lazily constructed from the tree state dict produced by the
        distributed training pipeline.  Exposes sklearn-compatible
        attributes (``children_left``, ``feature``, ``threshold``,
        ``value``, etc.).  See :class:`~maxframe.learn.tree.Tree` for
        details.
        """
        if not hasattr(self, "tree_model_"):
            raise AttributeError(
                "'{}' object has no attribute 'tree_'. "
                "Call 'fit' first.".format(type(self).__name__)
            )
        if not hasattr(self, "_tree_obj") or self._tree_obj is None:
            tree_state_data = self.tree_model_.fetch()
            self._tree_obj = Tree.from_state_dict(tree_state_data)
        return self._tree_obj

    def predict(self, X, execute=False, session=None, run_kwargs=None):
        """Predict class for X.

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
        check_is_fitted(self)
        X = mt.tensor(X, dtype=np.float64)
        op = DecisionTreePredictOp(
            tree_state=self.tree_model_,
            n_classes=self.n_classes_,
            is_classification=True,
        )
        labels, _proba = op(X)
        # Map 0-based indices back to original class labels
        classes = mt.tensor(self.classes_)
        res = classes[labels]
        if execute:
            res.execute(session=session, **(run_kwargs or {}))
        return res

    def predict_proba(self, X, execute=False, session=None, run_kwargs=None):
        """Predict class probabilities for X.

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
        check_is_fitted(self)
        X = mt.tensor(X, dtype=np.float64)
        op = DecisionTreePredictOp(
            tree_state=self.tree_model_,
            n_classes=self.n_classes_,
            is_classification=True,
        )
        _labels, proba = op(X)
        if execute:
            proba.execute(session=session, **(run_kwargs or {}))
        return proba

    def _validate_params(self):
        """Validate estimator parameters."""
        if self.criterion not in ("gini", "entropy"):
            raise ValueError(
                f"Criterion '{self.criterion}' is not supported. "
                f"Valid criteria are 'gini' and 'entropy'."
            )
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError(
                f"max_depth must be greater than zero or None. "
                f"Got {self.max_depth}."
            )
        if self.max_bins < 2:
            raise ValueError(f"max_bins must be at least 2. Got {self.max_bins}.")


class DecisionTreeRegressor(RegressorMixin, BaseEstimator):
    """A distributed decision tree regressor.

    This estimator implements a distributed version of the decision tree
    regressor using approximate split finding with feature binning,
    following the approach described in [1]_. The algorithm is designed
    for large-scale datasets that do not fit in a single machine's memory
    and runs on MaxCompute's distributed execution engine.

    Unlike scikit-learn's DecisionTreeRegressor which uses an exact
    greedy algorithm to find optimal split points, this implementation
    discretizes continuous features into bins and finds approximate splits.
    This trades a small amount of split quality for the ability to
    efficiently aggregate statistics across distributed partitions.

    Parameters
    ----------
    criterion : {"squared_error"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes L2
        loss using the mean of each terminal node.

        .. note::
            "absolute_error", "friedman_mse", and "poisson" criteria are
            not yet supported in the distributed implementation.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded
        until all leaves contain less than min_samples_split samples.
        Note that very deep trees may require many distributed
        communication rounds.

    max_bins : int, default=32
        The maximum number of bins used to discretize continuous features.
        Higher values improve split approximation quality at the cost of
        increased memory usage and communication overhead. This parameter
        has no equivalent in scikit-learn and is specific to the
        distributed binning algorithm.

        Memory usage per node group is approximately
        ``n_features * max_bins * 3 * 8`` bytes (3 doubles per bin for
        regression: count, sum, squared sum). When the memory budget
        (controlled by ``max_memory_in_mb``) is exceeded, fewer nodes are
        processed per iteration, increasing the number of data passes.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves
        at least ``min_samples_leaf`` training samples in each of the left
        and right branches.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, {"sqrt", "log2"} or None, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features))` features are considered.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. To obtain a deterministic
        behaviour during fitting, ``random_state`` has to be fixed to an
        integer.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the
        impurity greater than or equal to this value.

    max_memory_in_mb : int, default=256
        The maximum memory in MB to use for aggregating statistics for a
        group of nodes during training. This parameter has no equivalent
        in scikit-learn and is specific to the distributed algorithm.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.

    max_features_ : int
        The inferred value of max_features.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit. Defined only when `X` has
        feature names that are all strings.

    n_outputs_ : int
        The number of outputs when fit is performed.

    tree_ : Tree instance
        The underlying Tree object.

    See Also
    --------
    DecisionTreeClassifier : A distributed decision tree classifier.
    RandomForestRegressor : A distributed random forest regressor.

    Notes
    -----
    **Distributed Algorithm**

    Same level-wise binning algorithm as
    :class:`~maxframe.learn.tree.DecisionTreeClassifier`, but with
    regression-specific sufficient statistics: for each bin, the aggregated
    count, weighted sum, and weighted sum of squares are tracked. The
    squared_error criterion computes variance reduction at each candidate
    split.

    **Limitations compared to scikit-learn**

    In addition to the limitations documented in
    :class:`~maxframe.learn.tree.DecisionTreeClassifier`:

    - Only "squared_error" criterion is supported. "absolute_error",
      "friedman_mse", and "poisson" criteria are not yet available in
      the distributed implementation.
    - ``class_weight`` is not applicable to regressors.

    References
    ----------

    .. [1] L. Breiman, J. Friedman, R. Olshen, and C. Stone,
           "Classification and Regression Trees", Wadsworth, 1984.

    Examples
    --------
    >>> from maxframe.learn.tree import DecisionTreeRegressor
    >>> reg = DecisionTreeRegressor(max_depth=5, max_bins=32)
    >>> reg.fit(X_train, y_train)
    DecisionTreeRegressor(max_bins=32, max_depth=5)
    >>> predictions = reg.predict(X_test)
    """

    def __init__(
        self,
        *,
        criterion="squared_error",
        max_depth=None,
        max_bins=32,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        min_impurity_decrease=0.0,
        max_memory_in_mb=256,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.max_memory_in_mb = max_memory_in_mb

    # Explicit data attributes to prevent _get_data_attributes() from
    # scanning all public attributes ending with '_', which would trigger
    # the lazy ``tree_`` property and call ``tree_model_.fetch()`` on
    # unexecuted tileables.
    _data_attributes = ["tree_model_", "labels_", "feature_importances_", "_metadata"]

    def fit(
        self, X, y, sample_weight=None, execute=False, session=None, run_kwargs=None
    ):
        """Fit a distributed decision tree regressor.

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
        self : DecisionTreeRegressor
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
        # Convert sample_weight to tensor if provided
        if sample_weight is not None:
            sample_weight = mt.tensor(sample_weight, dtype=np.float64)
        self.n_features_ = n_features
        op = DecisionTreeFitOp(
            criterion=self.criterion,
            max_depth=self.max_depth,
            max_bins=self.max_bins,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            min_impurity_decrease=self.min_impurity_decrease,
            max_memory_in_mb=self.max_memory_in_mb,
            is_classification=False,
            n_classes=1,
            n_features=self.n_features_,
        )
        tree_model, labels, feature_importances, metadata = op(X, y, sample_weight)
        self.tree_model_ = tree_model
        self.labels_ = labels
        self.feature_importances_ = feature_importances
        self._metadata = metadata
        # Invalidate cached tree representation
        self._tree_obj = None
        if execute:
            execute_tileables(
                self.tree_model_,
                self.labels_,
                self.feature_importances_,
                self._metadata,
                session=session,
                **(run_kwargs or {}),
            )
        return self

    @property
    def tree_(self):
        """The underlying Tree object.

        Lazily constructed from the tree state dict produced by the
        distributed training pipeline.  Exposes sklearn-compatible
        attributes (``children_left``, ``feature``, ``threshold``,
        ``value``, etc.).  See :class:`~maxframe.learn.tree.Tree` for
        details.
        """
        if not hasattr(self, "tree_model_"):
            raise AttributeError(
                "'{}' object has no attribute 'tree_'. "
                "Call 'fit' first.".format(type(self).__name__)
            )
        if not hasattr(self, "_tree_obj") or self._tree_obj is None:
            tree_state_data = self.tree_model_.fetch()
            self._tree_obj = Tree.from_state_dict(tree_state_data)
        return self._tree_obj

    def predict(self, X, execute=False, session=None, run_kwargs=None):
        """Predict regression value for X.

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
        check_is_fitted(self)
        X = mt.tensor(X, dtype=np.float64)
        op = DecisionTreePredictOp(
            tree_state=self.tree_model_,
            is_classification=False,
        )
        predictions, _ = op(X)
        if execute:
            predictions.execute(session=session, **(run_kwargs or {}))
        return predictions

    def _validate_params(self):
        """Validate estimator parameters."""
        if self.criterion != "squared_error":
            raise ValueError(
                f"Criterion '{self.criterion}' is not supported. "
                f"Only 'squared_error' is currently supported in the "
                f"distributed implementation."
            )
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError(
                f"max_depth must be greater than zero or None. "
                f"Got {self.max_depth}."
            )
        if self.max_bins < 2:
            raise ValueError(f"max_bins must be at least 2. Got {self.max_bins}.")
