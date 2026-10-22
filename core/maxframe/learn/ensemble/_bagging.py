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

"""Bootstrap and subsampling utilities for distributed random forests.

This module provides utilities for generating bootstrap sample weights
(Poisson-distributed) and subsample indices for random forest training.
These are computed per-partition to avoid data movement.
"""

import math

import numpy as np


def generate_bootstrap_samples(
    n_samples,
    n_estimators,
    random_state=None,
    max_samples=None,
):
    """Generate bootstrap sample weights for random forest training.

    For each tree, each sample gets a Poisson(1) weight, simulating
    bootstrap sampling without actually resampling the data. This is
    the approach used by Spark's random forest implementation.

    Parameters
    ----------
    n_samples : int
        Number of training samples.
    n_estimators : int
        Number of trees in the forest.
    random_state : int or RandomState, default=None
        Random state for reproducibility.
    max_samples : int or float, default=None
        The number of samples to draw for each tree.

        - If None, draw n_samples (standard bootstrap).
        - If int, draw max_samples samples.
        - If float, draw max(round(n_samples * max_samples), 1) samples.

    Returns
    -------
    sample_weights : ndarray of shape (n_samples, n_estimators)
        Per-tree sample weights. Each column contains Poisson-distributed
        weights for one tree.
    """
    rng = (
        np.random.RandomState(random_state)
        if not isinstance(random_state, np.random.RandomState)
        else random_state
    )

    if max_samples is not None:
        if isinstance(max_samples, float):
            if not 0.0 < max_samples <= 1.0:
                raise ValueError(
                    f"max_samples as a float must be in (0.0, 1.0], "
                    f"got {max_samples}."
                )
            # A float is already the global Bernoulli probability; use it
            # directly so the selection rate is independent of chunk size.
            # Rounding to a per-chunk integer count (max(round(...), 1))
            # would force every small chunk to select all its rows.
            p = max_samples
        else:
            if max_samples <= 0:
                raise ValueError(
                    f"max_samples as int must be positive, got {max_samples}."
                )
            # Integer max_samples is the desired sample count; convert to
            # the global Bernoulli probability. The tiler pre-converts
            # integers to a float probability (handled by the float branch
            # above), but a direct call with an int still reaches here and
            # must divide by n_samples to avoid p=1.0.
            p = min(max_samples / n_samples, 1.0)
        sample_weights = rng.binomial(1, p, size=(n_samples, n_estimators)).astype(
            np.float64
        )
    else:
        # Standard bootstrap: Poisson(1) weights
        sample_weights = rng.poisson(1.0, size=(n_samples, n_estimators)).astype(
            np.float64
        )

    return sample_weights


def compute_feature_subset_size(
    n_features,
    feature_subset_strategy="auto",
    is_classification=True,
):
    """Compute the number of features to consider at each split.

    Parameters
    ----------
    n_features : int
        Total number of features.
    feature_subset_strategy : {"auto", "sqrt", "log2", "onethird", "all"} \
            or int, default="auto"
        Strategy for determining the number of features.
    is_classification : bool, default=True
        Whether this is a classification task (affects "auto" strategy).

    Returns
    -------
    n_subset_features : int
        Number of features to consider at each split.
    """
    if isinstance(feature_subset_strategy, int):
        if feature_subset_strategy <= 0:
            raise ValueError(
                f"feature_subset_strategy as int must be positive, "
                f"got {feature_subset_strategy}."
            )
        return min(feature_subset_strategy, n_features)

    if isinstance(feature_subset_strategy, float):
        if not 0.0 < feature_subset_strategy <= 1.0:
            raise ValueError(
                f"feature_subset_strategy as float must be in (0.0, 1.0], "
                f"got {feature_subset_strategy}."
            )
        return max(1, int(feature_subset_strategy * n_features))

    strategy = feature_subset_strategy.lower()

    if strategy == "auto":
        if is_classification:
            return max(1, int(math.ceil(math.sqrt(n_features))))
        else:
            return max(1, int(math.ceil(n_features / 3)))
    elif strategy == "sqrt":
        return max(1, int(math.ceil(math.sqrt(n_features))))
    elif strategy == "log2":
        return max(1, int(math.ceil(math.log2(max(n_features, 2)))))
    elif strategy == "onethird":
        return max(1, int(math.ceil(n_features / 3)))
    elif strategy == "all":
        return n_features
    else:
        raise ValueError(
            f"Unsupported feature_subset_strategy: '{feature_subset_strategy}'. "
            f"Valid strategies are 'auto', 'sqrt', 'log2', 'onethird', 'all', "
            f"or a positive int/float."
        )


def generate_feature_subsets(
    n_features,
    n_estimators,
    feature_subset_strategy="auto",
    is_classification=True,
    random_state=None,
):
    """Generate per-tree feature subsets for random forest training.

    For each tree, selects a random subset of features to consider at
    each split point. The subset size is determined by
    ``feature_subset_strategy``.

    Parameters
    ----------
    n_features : int
        Total number of features.
    n_estimators : int
        Number of trees in the forest.
    feature_subset_strategy : str or int, default="auto"
        Strategy for determining the number of features per split.
    is_classification : bool, default=True
        Whether this is a classification task.
    random_state : int or RandomState, default=None
        Random state for reproducibility.

    Returns
    -------
    feature_indices : list of ndarray
        List of length n_estimators, each entry is an array of feature
        indices to consider for that tree.
    """
    n_subset = compute_feature_subset_size(
        n_features, feature_subset_strategy, is_classification
    )

    rng = (
        np.random.RandomState(random_state)
        if not isinstance(random_state, np.random.RandomState)
        else random_state
    )
    feature_indices = []

    for _ in range(n_estimators):
        if n_subset >= n_features:
            indices = np.arange(n_features, dtype=np.intp)
        else:
            indices = rng.choice(n_features, size=n_subset, replace=False)
            indices.sort()
        feature_indices.append(indices)

    return feature_indices


def generate_feature_subsets_per_node(
    n_features,
    n_subset,
    n_estimators,
    node_ids_per_tree,
    random_state=None,
):
    """Generate per-node feature subsets for random forest training.

    Unlike :func:`generate_feature_subsets` which draws one subset per tree
    and reuses it at every node, this function draws a distinct subset
    for each (tree, node) pair using a deterministic seed derived from
    (random_state, tree_idx, node_id).  This matches Spark's RandomForest
    behavior where a new feature subset is selected at each node split.

    Parameters
    ----------
    n_features : int
        Total number of features.
    n_subset : int
        Number of features to select per node (output of
        :func:`compute_feature_subset_size`).
    n_estimators : int
        Number of trees in the forest.
    node_ids_per_tree : list of array-like
        Active node IDs for each tree at the current depth.
    random_state : int or RandomState, default=None
        Base random state for reproducibility.

    Returns
    -------
    feature_indices_per_node_per_tree : list of list of ndarray
        feature_indices_per_node_per_tree[tree_idx][node_pos] is the
        sorted ndarray of feature indices for that node.
    """
    if n_subset >= n_features:
        # No subsampling needed - all features for every node
        return [
            [np.arange(n_features, dtype=np.intp)] * len(node_ids)
            for node_ids in node_ids_per_tree
        ]

    # Extract a stable base seed once, independent of RNG consumption order,
    # so per-node seeds are deterministic regardless of node batching.
    if isinstance(random_state, np.random.RandomState):
        _state = random_state.get_state()
        base_seed = (
            int(_state[1][0]) if _state[0] == "MT19937" else hash(str(random_state))
        )
    else:
        base_seed = int(random_state) if random_state is not None else 0
    result = []
    for tree_idx, node_ids in enumerate(node_ids_per_tree):
        tree_subsets = []
        for node_id in node_ids:
            seed = (
                base_seed ^ (int(tree_idx) * 0x9E3779B1) ^ (int(node_id) * 0x85EBCA77)
            ) & 0xFFFFFFFF
            node_rng = np.random.RandomState(seed)
            indices = node_rng.choice(n_features, size=n_subset, replace=False)
            indices.sort()
            tree_subsets.append(indices)
        result.append(tree_subsets)
    return result
