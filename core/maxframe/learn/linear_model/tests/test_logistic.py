# Copyright 1998-2026 Alibaba Group Holding Ltd.
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

from maxframe.learn.linear_model import LogisticRegression


def test_invalid_penalty_raises():
    """fit() with an unsupported penalty value should raise ValueError."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    clf = LogisticRegression(penalty="bogus", solver="lbfgs")
    with pytest.raises(ValueError, match="penalties"):
        clf.fit(X, y)


def test_elasticnet_without_l1_ratio_raises():
    """penalty='elasticnet' without l1_ratio should raise ValueError."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    clf = LogisticRegression(penalty="elasticnet", solver="lbfgs")
    with pytest.raises(ValueError, match="l1_ratio"):
        clf.fit(X, y)


def test_negative_C_raises():
    """Negative C should raise ValueError."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    clf = LogisticRegression(C=-1.0, solver="lbfgs")
    with pytest.raises(ValueError, match="Penalty term must be positive"):
        clf.fit(X, y)


def test_negative_max_iter_raises():
    """Negative max_iter should raise ValueError."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    clf = LogisticRegression(max_iter=-1, solver="lbfgs")
    with pytest.raises(
        ValueError, match="Maximum number of iteration must be positive"
    ):
        clf.fit(X, y)


def test_unsupported_solver_raises():
    """Unsupported solver should raise ValueError at fit() time."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    for solver in ["sag", "saga"]:
        with pytest.raises(ValueError, match="solvers"):
            LogisticRegression(solver=solver).fit(X, y)


@pytest.mark.parametrize("solver", ["newton-cg", "newton-cholesky"])
@pytest.mark.parametrize("invalid_penalty", ["l1", "elasticnet"])
def test_newton_solver_invalid_penalty_raises(solver, invalid_penalty):
    """Newton solvers should raise ValueError for l1/elasticnet penalties."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    clf = LogisticRegression(penalty=invalid_penalty, solver=solver)
    with pytest.raises(ValueError, match="supports only penalties"):
        clf.fit(X, y)


@pytest.mark.parametrize("solver", ["newton-cg", "newton-cholesky"])
def test_newton_solver_valid_penalties(solver):
    """Newton solvers should accept l2 and none penalties without error."""
    # X, y not needed - only testing validation at init
    for penalty in ["l2", "none"]:
        LogisticRegression(penalty=penalty, solver=solver)  # noqa: F841


@pytest.mark.parametrize("invalid_penalty", ["elasticnet", "none"])
def test_liblinear_solver_invalid_penalty_raises(invalid_penalty):
    """liblinear should raise ValueError for elasticnet/none penalties."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    clf = LogisticRegression(penalty=invalid_penalty, solver="liblinear")
    with pytest.raises(ValueError, match="supports only penalties"):
        clf.fit(X, y)


def test_liblinear_solver_valid_penalties():
    """liblinear should accept l1 and l2 penalties without error."""
    # X, y not needed - only testing validation at init
    for penalty in ["l1", "l2"]:
        LogisticRegression(penalty=penalty, solver="liblinear")  # noqa: F841
