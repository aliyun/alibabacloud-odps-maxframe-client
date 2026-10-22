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

import pickle

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from maxframe.learn.tree._tree import Tree

TREE_LEAF = -1
TREE_UNDEFINED = -2


# ---------------------------------------------------------------------------
# Helpers – build trees via from_state_dict (the public API)
# ---------------------------------------------------------------------------


def _cls_tree_state():
    """Depth-2 classification tree:

    [0] f=0, t=0.0 (100 samples, value=[50,30,20])
     ├── [1] f=1, t=0.5 (60 samples, value=[10,30,20])
     │    ├── [3] leaf (30, value=[5,25,0])
     │    └── [4] leaf (30, value=[5,5,20])
     └── [2] leaf (40, value=[40,0,0])
    """
    return {
        "n_features": 2,
        "n_classes": 3,
        "n_outputs": 1,
        "nodes": [
            {
                "is_leaf": False,
                "feature": 0,
                "threshold": 0.0,
                "left_child": 1,
                "right_child": 2,
                "impurity": 0.6667,
                "n_node_samples": 100,
                "weighted_n_node_samples": 100.0,
                "value": [50, 30, 20],
            },
            {
                "is_leaf": False,
                "feature": 1,
                "threshold": 0.5,
                "left_child": 3,
                "right_child": 4,
                "impurity": 0.5,
                "n_node_samples": 60,
                "weighted_n_node_samples": 60.0,
                "value": [10, 30, 20],
            },
            {
                "is_leaf": True,
                "impurity": 0.0,
                "n_node_samples": 40,
                "weighted_n_node_samples": 40.0,
                "value": [40, 0, 0],
            },
            {
                "is_leaf": True,
                "impurity": 0.0,
                "n_node_samples": 30,
                "weighted_n_node_samples": 30.0,
                "value": [5, 25, 0],
            },
            {
                "is_leaf": True,
                "impurity": 0.0,
                "n_node_samples": 30,
                "weighted_n_node_samples": 30.0,
                "value": [5, 5, 20],
            },
        ],
    }


def _reg_tree_state():
    """Depth-1 regression tree:

    [0] f=0, t=0.0 (100, value=[3.5])
     ├── [1] leaf (50, value=[2.0])
     └── [2] leaf (50, value=[5.0])
    """
    return {
        "n_features": 2,
        "n_classes": 1,
        "n_outputs": 1,
        "nodes": [
            {
                "is_leaf": False,
                "feature": 0,
                "threshold": 0.0,
                "left_child": 1,
                "right_child": 2,
                "impurity": 4.5,
                "n_node_samples": 100,
                "weighted_n_node_samples": 100.0,
                "value": [3.5],
            },
            {
                "is_leaf": True,
                "impurity": 0.0,
                "n_node_samples": 50,
                "weighted_n_node_samples": 50.0,
                "value": [2.0],
            },
            {
                "is_leaf": True,
                "impurity": 0.0,
                "n_node_samples": 50,
                "weighted_n_node_samples": 50.0,
                "value": [5.0],
            },
        ],
    }


# ===========================================================================
# Tree construction from state dict
# ===========================================================================


@pytest.mark.parametrize(
    "state,n_nodes,max_depth",
    [
        (_cls_tree_state(), 5, 2),
        (_reg_tree_state(), 3, 1),
    ],
)
def test_from_state_dict_node_count_and_depth(state, n_nodes, max_depth):
    tree = Tree.from_state_dict(state)
    assert tree.node_count == n_nodes
    assert tree.max_depth == max_depth


@pytest.mark.parametrize(
    "state,n_leaves",
    [
        (_cls_tree_state(), 3),
        (_reg_tree_state(), 2),
    ],
)
def test_from_state_dict_n_leaves(state, n_leaves):
    tree = Tree.from_state_dict(state)
    assert tree.n_leaves == n_leaves


@pytest.mark.parametrize(
    "n_classes_input,expected_max",
    [
        (2, 2),
        (5, 5),
        (np.array([3, 5]), 5),  # multi-output
    ],
)
def test_from_state_dict_max_n_classes(n_classes_input, expected_max):
    n_outputs = len(n_classes_input) if isinstance(n_classes_input, np.ndarray) else 1
    state = {
        "n_features": 1,
        "n_classes": n_classes_input,
        "n_outputs": n_outputs,
        "nodes": [{"is_leaf": True}],
    }
    tree = Tree.from_state_dict(state)
    assert tree.max_n_classes == expected_max


def test_from_state_dict_value_padding():
    """Values shorter than max_n_classes are zero-padded."""
    state = {
        "n_features": 1,
        "n_classes": 5,
        "n_outputs": 1,
        "nodes": [{"is_leaf": True, "value": [3, 7]}],
    }
    tree = Tree.from_state_dict(state)
    np.testing.assert_array_equal(tree.value[0, 0], [3, 7, 0, 0, 0])


def test_from_state_dict_multi_output_value():
    """Multi-output values survive round-trip through from_state_dict.

    With n_outputs=2 and n_classes=[2, 2], a leaf value of [[3, 1], [1, 3]]
    must be fully preserved — both output rows must round-trip, not just
    the first.
    """
    state = {
        "n_features": 1,
        "n_classes": [2, 2],
        "n_outputs": 2,
        "nodes": [{"is_leaf": True, "value": [[3, 1], [1, 3]]}],
    }
    tree = Tree.from_state_dict(state)
    assert tree.node_count == 1
    # tree.value has shape (node_count, n_outputs, max_n_classes)
    assert tree.value.shape == (1, 2, 2)
    np.testing.assert_array_equal(tree.value[0, 0], [3, 1])
    np.testing.assert_array_equal(tree.value[0, 1], [1, 3])


def test_from_state_dict_empty_nodes():
    tree = Tree.from_state_dict(
        {"n_features": 2, "n_classes": 2, "n_outputs": 1, "nodes": []}
    )
    assert tree.node_count == 0
    # All sklearn-compatible properties must return shape (0,) on an empty
    # tree rather than raising a missing-field error. The empty node array
    # uses the full Node structured dtype.
    assert tree.children_left.shape == (0,)
    assert tree.children_right.shape == (0,)
    assert tree.feature.shape == (0,)
    assert tree.threshold.shape == (0,)
    assert tree.impurity.shape == (0,)
    assert tree.n_node_samples.shape == (0,)
    assert tree.weighted_n_node_samples.shape == (0,)
    assert tree.value.shape == (0, 1, 2)


def test_from_state_dict_max_depth_chain():
    """Chain tree: root -> left -> left -> leaf. Max depth = 2."""
    state = {
        "n_features": 1,
        "n_classes": 2,
        "n_outputs": 1,
        "nodes": [
            {
                "is_leaf": False,
                "feature": 0,
                "threshold": 0.3,
                "left_child": 1,
                "right_child": 2,
            },
            {
                "is_leaf": False,
                "feature": 0,
                "threshold": 0.1,
                "left_child": 3,
                "right_child": 4,
            },
            {"is_leaf": True},
            {"is_leaf": True},
            {"is_leaf": True},
        ],
    }
    tree = Tree.from_state_dict(state)
    assert tree.max_depth == 2


# ===========================================================================
# apply – routing samples to correct leaves
# ===========================================================================


@pytest.mark.parametrize(
    "X,expected_leaves",
    [
        (np.array([[-1.0, 0.0]]), [3]),  # left-left
        (np.array([[1.0, 0.0]]), [2]),  # right
        (np.array([[-1.0, 1.0]]), [4]),  # left-right
        (np.array([[-1.0, 0.0], [1.0, 0.0], [-1.0, 1.0]]), [3, 2, 4]),
    ],
)
def test_apply_classification(X, expected_leaves):
    tree = Tree.from_state_dict(_cls_tree_state())
    np.testing.assert_array_equal(tree.apply(X), expected_leaves)


@pytest.mark.parametrize(
    "X,expected_leaves",
    [
        (np.array([[-1.0, 0.0]]), [1]),  # left leaf
        (np.array([[1.0, 0.0]]), [2]),  # right leaf
    ],
)
def test_apply_regression(X, expected_leaves):
    tree = Tree.from_state_dict(_reg_tree_state())
    np.testing.assert_array_equal(tree.apply(X), expected_leaves)


def test_apply_rejects_1d():
    tree = Tree.from_state_dict(_cls_tree_state())
    with pytest.raises(ValueError, match="2D"):
        tree.apply(np.array([1.0, 2.0]))


# ===========================================================================
# predict
# ===========================================================================


def test_predict_classification_returns_value_at_leaf():
    tree = Tree.from_state_dict(_cls_tree_state())
    # Left-left path -> node 3, predict returns value array at that leaf
    result = tree.predict(np.array([[-1.0, 0.0]]))
    assert result.shape == (1, 3)
    np.testing.assert_array_almost_equal(result[0], [5.0, 25.0, 0.0])


@pytest.mark.parametrize(
    "X,expected",
    [
        (np.array([[-1.0, 0.0]]), [[2.0]]),  # left leaf
        (np.array([[1.0, 0.0]]), [[5.0]]),  # right leaf
    ],
)
def test_predict_regression(X, expected):
    tree = Tree.from_state_dict(_reg_tree_state())
    np.testing.assert_array_almost_equal(tree.predict(X), expected)


def test_predict_rejects_1d():
    tree = Tree.from_state_dict(_cls_tree_state())
    with pytest.raises(ValueError, match="2D"):
        tree.predict(np.array([1.0, 2.0]))


# ===========================================================================
# decision_path
# ===========================================================================


@pytest.mark.parametrize(
    "X_row,expected_nnz",
    [
        (np.array([[-1.0, 0.0]]), 3),  # 0 -> 1 -> 3
        (np.array([[1.0, 0.0]]), 2),  # 0 -> 2
        (np.array([[-1.0, 1.0]]), 3),  # 0 -> 1 -> 4
    ],
)
def test_decision_path_nnz(X_row, expected_nnz):
    tree = Tree.from_state_dict(_cls_tree_state())
    path = tree.decision_path(X_row)
    assert isinstance(path, csr_matrix)
    assert path.shape == (1, 5)
    assert path.nnz == expected_nnz


def test_decision_path_visits_correct_nodes():
    tree = Tree.from_state_dict(_cls_tree_state())
    path = tree.decision_path(np.array([[-1.0, 0.0]])).toarray()[0]
    # Visits root(0) -> node1(1) -> leaf3(3)
    assert path[0] == 1
    assert path[1] == 1
    assert path[3] == 1
    assert path[2] == 0
    assert path[4] == 0


def test_decision_path_multiple_samples():
    tree = Tree.from_state_dict(_cls_tree_state())
    path = tree.decision_path(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    assert path.shape == (2, 5)


# ===========================================================================
# Feature importances
# ===========================================================================


def test_feature_importances_sum_to_one():
    tree = Tree.from_state_dict(_cls_tree_state())
    imp = tree.compute_feature_importances(normalize=True)
    assert imp.shape == (2,)
    assert imp.sum() == pytest.approx(1.0)


def test_feature_importances_nonnegative():
    tree = Tree.from_state_dict(_cls_tree_state())
    imp = tree.compute_feature_importances(normalize=False)
    assert all(imp >= 0)


def test_feature_importances_single_leaf_tree():
    tree = Tree(n_features=3, n_classes=np.array([2], dtype=np.intp), n_outputs=1)
    tree.add_node(
        TREE_UNDEFINED, False, True, TREE_UNDEFINED, TREE_UNDEFINED, 0.5, 10, 10.0
    )
    np.testing.assert_array_equal(tree.compute_feature_importances(), [0.0, 0.0, 0.0])


# ===========================================================================
# Serialization roundtrip
# ===========================================================================


@pytest.mark.parametrize("make_state", [_cls_tree_state, _reg_tree_state])
def test_pickle_roundtrip(make_state):
    tree = Tree.from_state_dict(make_state())
    tree2 = pickle.loads(pickle.dumps(tree))
    assert tree2.node_count == tree.node_count
    np.testing.assert_array_equal(tree2.children_left, tree.children_left)
    np.testing.assert_array_almost_equal(tree2.value, tree.value)


def test_setstate_dict_format():
    tree = Tree(n_features=2, n_classes=np.array([3], dtype=np.intp), n_outputs=1)
    tree.__setstate__(_cls_tree_state())
    assert tree.node_count == 5
    assert tree.children_left[0] == 1
