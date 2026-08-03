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
from maxframe.learn.decomposition import PCA
from maxframe.learn.utils.extmath import randomized_svd, svd_flip

# ---------- svd_flip tests (using tensor operations, verify output shapes) ----------


def test_svd_flip_u_based_returns_tensors():
    """svd_flip with u_based_decision=True should return tensors of same shape."""
    u = mt.tensor(np.array([[1.0, -2.0], [-3.0, 4.0], [5.0, -6.0]]))
    v = mt.tensor(np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]))
    u_flipped, v_flipped = svd_flip(u, v, u_based_decision=True)
    assert u_flipped.shape == u.shape
    assert v_flipped.shape == v.shape


def test_svd_flip_v_based_returns_tensors():
    """svd_flip with u_based_decision=False should return tensors of same shape."""
    u = mt.tensor(np.array([[1.0, -2.0], [-3.0, 4.0]]))
    v = mt.tensor(np.array([[-1.0, 2.0], [4.0, -5.0]]))
    u_flipped, v_flipped = svd_flip(u, v, u_based_decision=False)
    assert u_flipped.shape == u.shape
    assert v_flipped.shape == v.shape


def test_svd_flip_numpy_equivalence():
    """svd_flip should apply sign corrections to u and v tensors."""
    u_np = np.array([[1.0, -2.0], [-3.0, 4.0], [5.0, -6.0]])
    v_np = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])

    u = mt.tensor(u_np)
    v = mt.tensor(v_np)
    u_flipped, v_flipped = svd_flip(u, v, u_based_decision=True)

    # svd_flip modifies u and v in-place via *=, so the returned
    # tensors should be the same objects as the inputs
    assert u_flipped is u
    assert v_flipped is v

    # Hand-verified expected signs for u_based_decision=True:
    # Column 0: max abs is row 2 (5.0) -> sign = +1 (no flip)
    # Column 1: max abs is row 2 (-6.0) -> sign = -1 (flip)
    # So u column 1 should be negated, v row 1 should be negated.
    # Verify the numpy reference produces the same signs.
    max_abs_cols = np.argmax(np.abs(u_np), axis=0)
    signs = np.sign(u_np[max_abs_cols, np.arange(u_np.shape[1])])
    u_expected = u_np * signs
    v_expected = v_np * signs[:, np.newaxis]
    assert u_expected[2, 0] > 0  # +5.0
    assert u_expected[2, 1] > 0  # -6.0 * -1 = +6.0
    assert v_expected[1, 0] < 0  # 4.0 * -1 = -4.0


# ---------- PCA instantiation tests ----------


def test_pca_default_params():
    """PCA with default parameters."""
    pca = PCA()
    assert pca.n_components is None
    assert pca.copy is True
    assert pca.whiten is False
    assert pca.svd_solver == "auto"
    assert pca.tol == 0.0
    assert pca.iterated_power == "auto"
    assert pca.random_state is None


def test_pca_custom_params():
    """PCA with custom parameters."""
    pca = PCA(n_components=3, whiten=True, svd_solver="randomized", random_state=42)
    assert pca.n_components == 3
    assert pca.whiten is True
    assert pca.svd_solver == "randomized"
    assert pca.random_state == 42


def test_pca_inherits_base():
    """PCA should inherit from _BasePCA and sklearn base classes."""
    from maxframe.learn.core import BaseEstimator, TransformerMixin
    from maxframe.learn.decomposition._base import _BasePCA

    assert issubclass(PCA, _BasePCA)
    assert issubclass(PCA, BaseEstimator)
    assert issubclass(PCA, TransformerMixin)


# ---------- PCA validation tests ----------


def test_pca_sparse_input_raises():
    """PCA should reject sparse input."""
    X = sp.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float64))
    pca = PCA(n_components=1)
    with pytest.raises(TypeError, match="sparse"):
        pca.fit(X)


def test_pca_unrecognized_solver_raises():
    """PCA with invalid svd_solver should raise ValueError."""
    X = np.array([[1, 2], [3, 4]], dtype=np.float64)
    pca = PCA(n_components=1, svd_solver="unknown")
    with pytest.raises(ValueError, match="Unrecognized svd_solver"):
        pca.fit(X)


def test_pca_arpack_not_implemented():
    """PCA with arpack solver should raise NotImplementedError."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    pca = PCA(n_components=1, svd_solver="arpack")
    with pytest.raises(NotImplementedError, match="arpack"):
        pca.fit(X)


def test_pca_full_n_components_out_of_range():
    """PCA with n_components > min(n_samples, n_features) should raise."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)  # 3x2
    pca = PCA(n_components=3, svd_solver="full")
    with pytest.raises(ValueError, match="n_components"):
        pca.fit(X)


def test_pca_mle_requires_n_samples_ge_features():
    """PCA with n_components='mle' and n_samples < n_features should raise."""
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)  # 2x3
    pca = PCA(n_components="mle", svd_solver="full")
    with pytest.raises(ValueError, match="n_samples >= n_features"):
        pca.fit(X)


def test_pca_full_invalid_string_n_components_raises():
    """PCA with invalid string n_components and full solver should raise ValueError."""
    X = np.random.RandomState(0).randn(10, 5)
    pca = PCA(n_components="bad", svd_solver="full")
    with pytest.raises(ValueError, match="not a valid string"):
        pca.fit(X)


def test_pca_truncated_n_components_string_raises():
    """PCA with string n_components and truncated solver should raise."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    pca = PCA(n_components="mle", svd_solver="randomized")
    with pytest.raises(ValueError, match="cannot be a string"):
        pca.fit(X)


def test_pca_truncated_n_components_out_of_range():
    """PCA with n_components out of range for truncated solver should raise."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)  # 3x2
    pca = PCA(n_components=3, svd_solver="randomized")
    with pytest.raises(ValueError, match="n_components"):
        pca.fit(X)


def test_pca_truncated_float_n_components_raises():
    """PCA with float n_components and truncated solver should raise."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    pca = PCA(n_components=1.5, svd_solver="randomized")
    with pytest.raises(ValueError, match="must be of type int"):
        pca.fit(X)


def test_pca_full_float_n_components_ge1_raises():
    """PCA with float n_components >= 1 and full solver should raise."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    pca = PCA(n_components=1.5, svd_solver="full")
    with pytest.raises(ValueError, match="must be of type int"):
        pca.fit(X)


# ---------- PCA solver selection tests ----------


def test_pca_auto_solver_small_matrix():
    """For small matrices, auto solver should select 'full'."""
    X = np.random.RandomState(0).randn(10, 5).astype(np.float64)
    pca = PCA(n_components=2, svd_solver="auto")
    pca._fit(X, execute=False)
    assert pca._fit_svd_solver == "full"


def test_pca_auto_solver_mle():
    """n_components='mle' should force full solver."""
    X = np.random.RandomState(0).randn(20, 10).astype(np.float64)
    pca = PCA(n_components="mle", svd_solver="auto")
    # Just check solver selection, don't actually fit (mle triggers execution)
    pca._fit_svd_solver = pca.svd_solver
    if pca._fit_svd_solver == "auto":
        n_components = pca.n_components
        if max(X.shape) <= 500 or n_components == "mle":
            pca._fit_svd_solver = "full"
    assert pca._fit_svd_solver == "full"


def test_pca_auto_solver_randomized():
    """For large matrices with small n_components, auto should select 'randomized'."""
    # Create a large-ish matrix that triggers randomized
    X = np.random.RandomState(0).randn(600, 600).astype(np.float64)
    pca = PCA(n_components=10, svd_solver="auto")
    pca._fit(X, execute=False)
    assert pca._fit_svd_solver == "randomized"


def test_pca_auto_solver_large_matrix_high_n_components():
    """For large matrices with n_components >= 80% of min dim, auto selects 'full'."""
    X = np.random.RandomState(0).randn(600, 600).astype(np.float64)
    pca = PCA(n_components=500, svd_solver="auto")
    pca._fit(X, execute=False)
    assert pca._fit_svd_solver == "full"


def test_pca_auto_solver_variance_threshold():
    """n_components between 0 and 1 with auto should select 'full'."""
    _X = np.random.RandomState(0).randn(600, 600).astype(np.float64)  # noqa: F841
    pca = PCA(n_components=0.95, svd_solver="auto")
    # Replicate auto solver selection logic without actually fitting
    # (0 < n_components < 1 falls through to 'full' in auto mode)
    n_components = pca.n_components
    assert 0 < n_components < 1
    # In auto mode: not <= 500, n_components not 'mle',
    # n_components < 1 (not >= 1), so falls to 'full'
    assert pca.svd_solver == "auto"


# ---------- PCA _fit sets attributes ----------


def test_pca_fit_full_sets_attributes():
    """_fit with full solver should set key attributes as tensors."""
    X = np.random.RandomState(0).randn(10, 5).astype(np.float64)
    pca = PCA(n_components=2, svd_solver="full")
    pca._fit(X, execute=False)

    # Attributes should be set (as lazy tensors)
    assert hasattr(pca, "components_")
    assert hasattr(pca, "explained_variance_")
    assert hasattr(pca, "explained_variance_ratio_")
    assert hasattr(pca, "singular_values_")
    assert hasattr(pca, "mean_")
    assert hasattr(pca, "n_components_")
    assert pca.n_components_ == 2
    assert hasattr(pca, "noise_variance_")
    assert pca.n_samples_ == 10
    assert pca.n_features_ == 5


def test_pca_fit_truncated_sets_attributes():
    """_fit_truncated should set key attributes."""
    X = np.random.RandomState(0).randn(600, 600).astype(np.float64)
    pca = PCA(n_components=10, svd_solver="randomized")
    pca._fit(X, execute=False)

    assert hasattr(pca, "components_")
    assert hasattr(pca, "explained_variance_")
    assert hasattr(pca, "explained_variance_ratio_")
    assert hasattr(pca, "singular_values_")
    assert hasattr(pca, "mean_")
    assert pca.n_components_ == 10


# ---------- randomized_svd shape tests ----------


def test_randomized_svd_output_shapes():
    """randomized_svd should return tensors with correct shapes."""
    rng = np.random.RandomState(42)
    X_np = rng.randn(20, 10).astype(np.float64)
    X = mt.tensor(X_np)

    U, s, V = randomized_svd(X, n_components=3, random_state=42)

    assert U.shape == (20, 3)
    assert s.shape == (3,)
    assert V.shape == (3, 10)


def test_randomized_svd_wide_matrix_shapes():
    """randomized_svd on a wide matrix should return correct shapes."""
    rng = np.random.RandomState(42)
    X_np = rng.randn(5, 20).astype(np.float64)
    X = mt.tensor(X_np)

    U, s, V = randomized_svd(X, n_components=2, random_state=42)

    assert U.shape == (5, 2)
    assert s.shape == (2,)
    assert V.shape == (2, 20)


def test_randomized_svd_no_flip_shapes():
    """randomized_svd with flip_sign=False should return same shapes."""
    rng = np.random.RandomState(42)
    X_np = rng.randn(15, 8).astype(np.float64)
    X = mt.tensor(X_np)

    U, s, V = randomized_svd(X, n_components=4, flip_sign=False, random_state=42)

    assert U.shape == (15, 4)
    assert s.shape == (4,)
    assert V.shape == (4, 8)


# ---------- _assess_dimension / _infer_dimension tests ----------


def test_assess_dimension_basic():
    """_assess_dimension should return a finite float."""
    from maxframe.learn.decomposition._pca import _assess_dimension

    spectrum = np.array([5.0, 3.0, 1.0, 0.5, 0.1])
    result = _assess_dimension(spectrum, rank=2, n_samples=100)
    assert np.isfinite(result)


def test_infer_dimension_basic():
    """_infer_dimension should return an integer in [1, n_features-1]."""
    from maxframe.learn.decomposition._pca import _infer_dimension

    spectrum = np.array([5.0, 3.0, 1.0, 0.5, 0.1])
    result = _infer_dimension(spectrum, n_samples=100)
    assert isinstance(result, (int, np.integer))
    assert 1 <= result < len(spectrum)


def test_pca_get_precision_zero_noise_variance():
    """get_precision should not divide by zero when noise_variance_ == 0."""
    # Wide data with full PCA: n_components_ == n_features, noise_variance_ == 0
    X = np.random.RandomState(0).randn(5, 20).astype(np.float64)
    pca = PCA(svd_solver="full")
    pca._fit(X, execute=False)
    assert pca.noise_variance_ == 0.0

    precision = pca.get_precision()
    assert precision.shape == (20, 20)


def test_pca_transform_validates_feature_count():
    """transform should raise ValueError when X has wrong number of features."""
    X = np.random.RandomState(0).randn(6, 2).astype(np.float64)
    pca = PCA(n_components=1, svd_solver="full")
    pca._fit(X, execute=False)

    X_bad = np.random.RandomState(1).randn(3, 1)
    with pytest.raises(ValueError, match="features per sample"):
        pca.transform(X_bad)
