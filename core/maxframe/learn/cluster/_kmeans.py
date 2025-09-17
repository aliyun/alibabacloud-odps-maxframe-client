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

import warnings
from typing import List

import numpy as np

from ...tensor.core import TENSOR_TYPE

try:
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    ConvergenceWarning = RuntimeWarning

from ... import opcodes
from ... import tensor as mt
from ...core import ENTITY_TYPE, EntityData, ExecutableTuple, OutputType
from ...core.operator import Operator
from ...serialization.serializables import (
    AnyField,
    BoolField,
    Float64Field,
    Int32Field,
    KeyField,
    StringField,
)
from ...tensor.random import RandomStateField
from ..core import BaseEstimator, ClusterMixin, LearnOperatorMixin, TransformerMixin
from ..metrics.pairwise import euclidean_distances
from ..utils.validation import _check_sample_weight, check_array, check_is_fitted


def _validate_center_shape(X, n_centers, centers):
    """Check if centers is compatible with X and n_centers"""
    if len(centers) != n_centers:
        raise ValueError(
            "The shape of the initial centers (%s) "
            "does not match the number of clusters %i" % (centers.shape, n_centers)
        )
    if not np.isnan(X.shape[1]) and centers.shape[1] != X.shape[1]:
        raise ValueError(
            "The number of features of the initial centers %s "
            "does not match the number of features of the data %s."
            % (centers.shape[1], X.shape[1])
        )


class KMeansFitOp(LearnOperatorMixin, Operator):
    _op_type_ = opcodes.KMEANS_FIT

    X = KeyField("X")
    sample_weight = KeyField("sample_weight", default=None)
    n_clusters = Int32Field("n_clusters", default=None)
    init = AnyField("init", default=None)
    n_init = Int32Field("n_init", default=None)
    max_iter = Int32Field("max_iter", default=None)
    tol = Float64Field("tol", default=None)
    verbose = Int32Field("verbose", default=None)
    random_state = RandomStateField("random_state", default=None)
    copy_x = BoolField("copy_x", default=None)
    algorithm = StringField("algorithm", default=None)
    oversampling_factor = Int32Field("oversampling_factor", default=None)
    init_iter = Int32Field("init_iter", default=None)

    @property
    def output_limit(self) -> int:
        return 4

    @classmethod
    def _set_inputs(cls, op: "KMeansFitOp", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        input_iter = iter(op._inputs)
        op.X = next(input_iter)
        if isinstance(op.sample_weight, ENTITY_TYPE):
            op.sample_weight = next(input_iter)
        if isinstance(op.init, ENTITY_TYPE):
            op.init = next(input_iter)

    def __call__(self, X, sample_weight=None, init=None):
        self.X = X
        self.sample_weight = sample_weight

        inputs = [X]
        if isinstance(sample_weight, ENTITY_TYPE):
            inputs.append(sample_weight)
        if isinstance(init, ENTITY_TYPE):
            inputs.append(init)

        self._output_types = [OutputType.tensor] * 2 + [OutputType.scalar] * 2
        kws = [
            {
                "dtype": np.dtype(float),
                "shape": (np.nan, X.shape[1]),
            },  # cluster_centers
            {"dtype": np.dtype(int), "shape": (X.shape[0],)},  # labels
            {"dtype": np.dtype(float), "shape": ()},  # inertia
            {"dtype": np.dtype(int), "shape": ()},  # n_iter
        ]
        return self.new_tileables(inputs, kws=kws)

    def _check_params(self):
        # n_init
        if self.n_init <= 0:
            raise ValueError(f"n_init should be > 0, got {self.n_init} instead.")

        # max_iter
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        # n_clusters
        if not np.isnan(self.X.shape[0]) and self.X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={self.X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        # algorithm
        if self.algorithm not in ("auto", "full", "elkan"):
            raise ValueError(
                f"Algorithm must be 'auto', 'full' or 'elkan', "
                f"got {self.algorithm} instead."
            )

        if self.algorithm == "auto":
            # note:
            # Different from scikit-learn,
            # for now, full seems more efficient when data is large,
            # elkan needs to be tuned more
            # old: algorithm = "full" if self.n_clusters == 1 else "elkan"
            self.algorithm = "full"
        if self.algorithm == "elkan" and self.n_clusters == 1:
            warnings.warn(
                "algorithm='elkan' doesn't make sense for a single "
                "cluster. Using 'full' instead.",
                RuntimeWarning,
            )
            self.algorithm = "full"

        # init
        # fixme remove when callable init supported
        if callable(self.init):
            raise NotImplementedError("Callable init param not implemented by now")

        if not (
            hasattr(self.init, "__array__")
            or isinstance(self.init, TENSOR_TYPE)
            or callable(self.init)
            or (
                isinstance(self.init, str)
                and self.init in ["k-means++", "k-means||", "random"]
            )
        ):
            raise ValueError(
                f"init should be either 'k-means++'， 'k-mean||', 'random', "
                f"a tensor, a ndarray or a "
                f"callable, got '{self.init}' instead."
            )

        if hasattr(self.init, "__array__") and self.n_init != 1:
            warnings.warn(
                f"Explicit initial center position passed: performing only"
                f" one init in {self.__class__.__name__} instead of "
                f"n_init={self.n_init}.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.n_init = 1


class KMeansPredictOp(LearnOperatorMixin, Operator):
    _op_type_ = opcodes.KMEANS_PREDICT

    cluster_centers = KeyField("cluster_centers")
    X = KeyField("X")
    sample_weight = KeyField("sample_weight", default=None)

    @property
    def output_limit(self) -> int:
        return 2

    @classmethod
    def _set_inputs(cls, op: "KMeansPredictOp", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.cluster_centers = op.inputs[0]
        op.X = op.inputs[1]
        if op.sample_weight is not None:
            op.sample_weight = op.inputs[2]

    def __call__(self, cluster_centers, X, sample_weight=None):
        self.cluster_centers = cluster_centers
        self.X = X
        self.sample_weight = sample_weight

        inputs = [cluster_centers, X]
        if sample_weight is not None:
            inputs.append(sample_weight)

        self._output_types = [OutputType.tensor, OutputType.scalar]
        kws = [
            {"dtype": np.dtype(int), "shape": (X.shape[0],)},
            {"dtype": np.dtype(float), "shape": ()},
        ]
        return self.new_tileables(inputs, kws=kws)


class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """K-Means clustering.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'k-means||', 'random'} or tensor of shape \
            (n_clusters, n_features), default='k-means||'
        Method for initialization, defaults to 'k-means||':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'k-means||': scalable k-means++.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If a tensor is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    n_init : int, default=1
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True (default), then the original data is
        not modified, ensuring X is C-contiguous.  If False, the original data
        is modified, and put back before the function returns, but small
        numerical differences may be introduced by subtracting and then adding
        the data mean, in this case it will also not ensure that data is
        C-contiguous which may cause a significant slowdown.

    algorithm : {"auto", "full", "elkan"}, default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.

    oversampling_factor: int, default=2
        Only work for kmeans||, used in each iteration in kmeans||.

    init_iter: int, default=5
        Only work for kmeans||, indicates how may iterations required.

    Attributes
    ----------
    cluster_centers_ : tensor of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : tensor of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.

    See Also
    --------

    MiniBatchKMeans
        Alternative online implementation that does incremental updates
        of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.

    Notes
    -----
    The k-means problem is solved using either Lloyd's or Elkan's algorithm.

    The average complexity is given by O(k n T), were n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
    'How slow is the k-means method?' SoCG2006)

    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    If the algorithm stops before fully converging (because of ``tol`` or
    ``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
    i.e. the ``cluster_centers_`` will not be the means of the points in each
    cluster. Also, the estimator will reassign ``labels_`` after the last
    iteration to make ``labels_`` consistent with ``predict`` on the training
    set.

    Examples
    --------

    >>> from maxframe.learn.cluster import KMeans
    >>> import maxframe.tensor as mt
    >>> X = mt.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0, init='k-means++').fit(X).execute()
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]]).execute()
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means||",
        n_init=1,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="auto",
        oversampling_factor=2,
        init_iter=5,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = (
            random_state
            if isinstance(random_state, np.random.RandomState)
            else np.random.RandomState(random_state)
        )
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.oversampling_factor = oversampling_factor
        self.init_iter = init_iter

    def _check_test_data(self, X):
        X = check_array(
            X,
            accept_sparse=True,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:  # pragma: no cover
            raise ValueError(
                f"Incorrect number of features. Got {n_features} features, "
                f"expected {expected_n_features}"
            )

        return X

    def fit(
        self,
        X,
        y=None,
        sample_weight=None,
        execute=False,
        session=None,
        run_kwargs=None,
    ):
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self
            Fitted estimator.
        """
        expect_chunk_size_on_columns = mt.tensor(X).shape[1]
        if not np.isnan(expect_chunk_size_on_columns):
            X = mt.tensor(X, chunk_size={1: expect_chunk_size_on_columns})

        X = self._validate_data(
            X,
            accept_sparse=True,
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        op = KMeansFitOp(
            X=X,
            init=self.init,
            sample_weight=sample_weight,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            verbose=self.verbose,
            random_state=self.random_state,
            copy_x=self.copy_x,
            algorithm=self.algorithm,
            oversampling_factor=self.oversampling_factor,
            init_iter=self.init_iter,
        )

        # check params beforehand to raise errors early
        op._check_params()
        self.algorithm = op.algorithm
        self.n_init = op.n_init

        # Validate init array
        init = self.init
        if hasattr(init, "__array__"):
            init = check_array(init, dtype=X.dtype.type, copy=True, order="C")
            _validate_center_shape(X, self.n_clusters, init)

        [
            self.cluster_centers_,
            self.labels_,
            self.inertia_,
            self.n_iter_,
        ] = op(X, sample_weight=sample_weight, init=init)
        if execute:
            self.execute(session=session, run_kwargs=run_kwargs)
        return self

    def fit_predict(
        self,
        X,
        y=None,
        execute=False,
        sample_weight=None,
        session=None,
        run_kwargs=None,
    ):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(
            X,
            execute=execute,
            sample_weight=sample_weight,
            session=session,
            run_kwargs=run_kwargs,
        ).labels_

    def fit_transform(
        self, X, y=None, sample_weight=None, session=None, run_kwargs=None
    ):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        X_new : array of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        # Currently, this just skips a copy of the data if it is not in
        # np.array or CSR format already.
        # XXX This skips _check_test_data, which may change the dtype;
        # we should refactor the input validation.
        self.fit(X, sample_weight=sample_weight)
        return self._transform(X, session=session, run_kwargs=run_kwargs)

    def transform(self, X, session=None, run_kwargs=None):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : tensor of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        return self._transform(X, session=session, run_kwargs=run_kwargs)

    def _transform(self, X, session=None, run_kwargs=None):
        """guts of transform method; no input validation"""
        return euclidean_distances(X, self.cluster_centers_).execute(
            session=session, **(run_kwargs or dict())
        )

    def predict(
        self, X, sample_weight=None, execute=False, session=None, run_kwargs=None
    ):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : tensor of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)

        op = KMeansPredictOp()
        tp = ExecutableTuple(op(self.cluster_centers_, X, sample_weight))
        if execute:
            tp = tp.execute(session=session, **(run_kwargs or dict()))
        return tp[0]

    def score(
        self,
        X,
        y=None,
        execute=False,
        sample_weight=None,
        session=None,
        run_kwargs=None,
    ):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)

        op = KMeansPredictOp()
        tp = ExecutableTuple(op(self.cluster_centers_, X, sample_weight))
        if execute:
            tp = tp.execute(session=session, **(run_kwargs or dict()))
        return tp[1]


def k_means(
    X,
    n_clusters,
    sample_weight=None,
    init="k-means||",
    n_init=10,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    algorithm="auto",
    oversampling_factor=2,
    init_iter=5,
    return_n_iter=False,
):
    """K-means clustering algorithm.

    Parameters
    ----------
    X : Tensor, shape (n_samples, n_features)
        The observations to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory copy
        if the given data is not C-contiguous.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    sample_weight : array-like, shape (n_samples,), optional
        The weights for each observation in X. If None, all observations
        are assigned equal weight (default: None)

    init : {'k-means++', 'k-means||', 'random', or tensor, or a callable}, optional
        Method for initialization, default to 'k-means||':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'k-means||': scalable k-means++.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : boolean, optional
        Verbosity mode.

    tol : float, optional
        The relative increment in the results before declaring convergence.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, optional
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True (default), then the original data is
        not modified, ensuring X is C-contiguous.  If False, the original data
        is modified, and put back before the function returns, but small
        numerical differences may be introduced by subtracting and then adding
        the data mean, in this case it will also not ensure that data is
        C-contiguous which may cause a significant slowdown.

    algorithm : "auto", "full" or "elkan", default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.

    oversampling_factor: int, default=2
        Only work for kmeans||, used in each iteration in kmeans||.

    init_iter: int, default=5
        Only work for kmeans||, indicates how may iterations required.

    return_n_iter : bool, optional
        Whether or not to return the number of iterations.

    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    best_n_iter : int
        Number of iterations corresponding to the best results.
        Returned only if `return_n_iter` is set to True.
    """

    est = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        tol=tol,
        random_state=random_state,
        copy_x=copy_x,
        algorithm=algorithm,
        oversampling_factor=oversampling_factor,
        init_iter=init_iter,
    ).fit(X, sample_weight=sample_weight)
    if return_n_iter:
        return ExecutableTuple(
            (est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_)
        )
    else:
        return ExecutableTuple((est.cluster_centers_, est.labels_, est.inertia_))
