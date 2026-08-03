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
import scipy.sparse as sp

import maxframe.tensor as mt
from maxframe.learn.core import BaseEstimator, TransformerMixin
from maxframe.learn.decomposition import TruncatedSVD

# ---------- instantiation tests ----------


def test_truncated_svd_default_params():
    """TruncatedSVD should expose the documented defaults."""
    tsvd = TruncatedSVD()
    assert tsvd.n_components == 2
    assert tsvd.algorithm == "randomized"
    assert tsvd.n_iter == 5
    assert tsvd.random_state is None
    assert tsvd.tol == 0.0


def test_truncated_svd_custom_params():
    """TruncatedSVD should store custom parameters without validation."""
    tsvd = TruncatedSVD(
        n_components=7, algorithm="randomized", n_iter=9, random_state=1, tol=1e-4
    )
    assert tsvd.n_components == 7
    assert tsvd.n_iter == 9
    assert tsvd.random_state == 1
    assert tsvd.tol == 1e-4


def test_truncated_svd_inherits_base_classes():
    """TruncatedSVD should inherit from BaseEstimator and TransformerMixin."""
    assert issubclass(TruncatedSVD, BaseEstimator)
    assert issubclass(TruncatedSVD, TransformerMixin)
    # It must NOT inherit the PCA base (no covariance/precision machinery).
    from maxframe.learn.decomposition._base import _BasePCA

    assert not issubclass(TruncatedSVD, _BasePCA)


# ---------- validation tests ----------


def test_truncated_svd_arpack_not_implemented():
    """TruncatedSVD with algorithm='arpack' should raise NotImplementedError."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    tsvd = TruncatedSVD(n_components=1, algorithm="arpack")
    with pytest.raises(NotImplementedError, match="arpack"):
        tsvd.fit(X)


def test_truncated_svd_unknown_algorithm_raises():
    """TruncatedSVD with an unknown algorithm should raise ValueError."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    tsvd = TruncatedSVD(n_components=1, algorithm="unknown")
    with pytest.raises(ValueError, match="unknown algorithm"):
        tsvd.fit(X)


def test_truncated_svd_n_components_gt_n_features_raises():
    """n_components must be <= n_features for the randomized solver."""
    X = np.random.RandomState(0).randn(20, 10)
    tsvd = TruncatedSVD(n_components=11, algorithm="randomized")
    with pytest.raises(ValueError, match="n_components must be <= n_features"):
        tsvd.fit(X)


def test_truncated_svd_n_components_eq_n_features_allowed():
    """n_components == n_features should be allowed for the randomized solver."""
    X = np.random.RandomState(0).randn(20, 10)
    tsvd = TruncatedSVD(n_components=10, algorithm="randomized", random_state=42)
    tsvd.fit(X, execute=False)
    assert tsvd.components_.shape == (10, 10)


def test_truncated_svd_n_components_non_integer_raises():
    """n_components must be an integer; floats and strings should be rejected."""
    X = np.random.RandomState(0).randn(20, 10)
    for n_components in (1.5, "5", None):
        tsvd = TruncatedSVD(n_components=n_components, algorithm="randomized")
        with pytest.raises(ValueError, match="must be of type int"):
            tsvd.fit(X)


def test_truncated_svd_n_components_lt_one_raises():
    """n_components must be >= 1."""
    X = np.random.RandomState(0).randn(20, 10)
    for n_components in (0, -1):
        tsvd = TruncatedSVD(n_components=n_components, algorithm="randomized")
        with pytest.raises(ValueError, match="must be >= 1"):
            tsvd.fit(X)


def test_truncated_svd_ensure_min_features():
    """A single feature should be rejected by check_array (ensure_min_features=2)."""
    X = np.array([[1.0], [2.0], [3.0]])
    tsvd = TruncatedSVD(n_components=1, algorithm="randomized")
    with pytest.raises(ValueError):
        tsvd.fit(X)


# ---------- attribute / shape tests (lazy, no execution) ----------


def test_truncated_svd_fit_sets_attributes():
    """fit_transform should set all four computed attributes as lazy tensors."""
    X = np.random.RandomState(0).randn(30, 12).astype(np.float64)
    tsvd = TruncatedSVD(n_components=5, random_state=42)
    X_transformed = tsvd.fit_transform(X, execute=False)

    for attr in (
        "components_",
        "explained_variance_",
        "explained_variance_ratio_",
        "singular_values_",
    ):
        assert hasattr(tsvd, attr)

    # TruncatedSVD does not set the PCA-only attributes.
    for attr in ("mean_", "noise_variance_", "n_components_", "n_samples_"):
        assert not hasattr(tsvd, attr)

    assert tsvd.components_.shape == (5, 12)
    assert tsvd.explained_variance_.shape == (5,)
    assert tsvd.explained_variance_ratio_.shape == (5,)
    assert tsvd.singular_values_.shape == (5,)
    assert X_transformed.shape == (30, 5)


def test_truncated_svd_fit_returns_self():
    """fit should return the estimator itself."""
    X = np.random.RandomState(0).randn(30, 12)
    tsvd = TruncatedSVD(n_components=4, random_state=42)
    assert tsvd.fit(X, execute=False) is tsvd


def test_truncated_svd_accepts_sparse():
    """TruncatedSVD should accept csr/csc sparse input (unlike PCA)."""
    X = sp.csr_matrix(np.random.RandomState(0).randn(30, 12))
    tsvd = TruncatedSVD(n_components=4, random_state=42)
    tsvd.fit_transform(X, execute=False)
    assert tsvd.components_.shape == (4, 12)


def test_truncated_svd_transform_shapes():
    """transform / inverse_transform should produce the right shapes (lazy)."""
    X = np.random.RandomState(0).randn(30, 12)
    tsvd = TruncatedSVD(n_components=4, random_state=42)
    tsvd.fit(X, execute=False)

    X_trans = tsvd.transform(X, execute=False)
    assert X_trans.shape == (30, 4)

    X_inv = tsvd.inverse_transform(X_trans, execute=False)
    assert X_inv.shape == (30, 12)


def test_truncated_svd_transform_validates_feature_count():
    """transform should reject inputs whose feature count differs from the fit data."""
    X = np.random.RandomState(0).randn(30, 12)
    tsvd = TruncatedSVD(n_components=4, random_state=42)
    tsvd.fit(X, execute=False)

    X_bad = np.random.RandomState(1).randn(10, 9)
    with pytest.raises(ValueError):
        tsvd.transform(X_bad, execute=False)


# ---------- variance computation (lazy graph sanity) ----------


def test_truncated_svd_explained_variance_ratio_graph():
    """explained_variance_ratio_ is built from the uncentered variance of X.

    Builds the lazy graph and checks the shapes are consistent with the
    ddof=0, no-centering formula (unlike PCA which centers and uses ddof=1).
    """
    X = np.random.RandomState(0).randn(25, 8)
    tsvd = TruncatedSVD(n_components=3, random_state=42)
    tsvd.fit_transform(X, execute=False)

    # The ratio tensor should be element-wise exp_var / var(X, axis=0).sum().
    # We only assert shape/dtype wiring here; the DPE test checks numeric value.
    assert tsvd.explained_variance_ratio_.shape == (3,)
    # singular_values_ feeds explained_variance_ via U*Sigma; shapes must match.
    assert tsvd.singular_values_.shape == tsvd.explained_variance_.shape


# ---------- randomized_svd integration (lazy shapes) ----------


def test_truncated_svd_randomized_svd_shapes():
    """TruncatedSVD should drive randomized_svd to the expected output shapes."""
    rng = np.random.RandomState(42)
    X_np = rng.randn(40, 15).astype(np.float64)
    X = mt.tensor(X_np)

    tsvd = TruncatedSVD(n_components=6, n_iter=4, random_state=42)
    X_trans = tsvd.fit_transform(X, execute=False)

    assert X_trans.shape == (40, 6)
    assert tsvd.components_.shape == (6, 15)
    assert tsvd.singular_values_.shape == (6,)
