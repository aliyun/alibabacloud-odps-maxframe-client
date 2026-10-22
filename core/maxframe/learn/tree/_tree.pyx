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

"""Cython implementation of the decision tree data structure.

This module provides the core Tree class used by both DecisionTree and
RandomForest estimators in MaxFrame's distributed learn module. The tree
is stored as a flat array of Node structs for efficient memory layout,
fast prediction, and compact serialization.

The Tree class also exposes sklearn-compatible array attributes
(children_left, feature, threshold, value, etc.) so that users can
inspect fitted trees via ``estimator.tree_`` just like scikit-learn.
"""

import numpy as np
from collections import deque

cimport numpy as cnp
from libc.math cimport isnan
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset

from maxframe.learn.tree._tree_utils cimport safe_realloc

# The following are defined in _tree.pxd and automatically available:
#   intp_t, uint8_t, Node struct, TREE_LEAF, TREE_UNDEFINED

cnp.import_array()

# Sentinel values (declared in .pxd, initialized here)
TREE_LEAF = -1
TREE_UNDEFINED = -2


cdef class Tree:
    """The underlying tree structure for a decision tree model.

    The tree is represented as a flat array of Node structs and a flat
    array of prediction values. This design enables:

    - O(1) node access via array indexing
    - Efficient serialization for distributed transfer
    - Fast prediction via tight C loops with GIL release
    - Memory-efficient storage with no per-node Python object overhead

    Additionally, the class exposes sklearn-compatible array attributes
    (``children_left``, ``feature``, ``threshold``, ``value``, etc.)
    and can be constructed from MaxFrame's internal dict-based state via
    the ``from_state_dict`` classmethod.
    """

    def __cinit__(self, intp_t n_features, cnp.ndarray n_classes,
                  intp_t n_outputs):
        """Initialize the tree.

        Parameters
        ----------
        n_features : int
            Number of features in the training data.
        n_classes : ndarray of shape (n_outputs,)
            Number of classes for each output.
        n_outputs : int
            Number of outputs (targets).
        """
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.max_n_classes = <intp_t>np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        # Copy n_classes into a C array
        cdef int i
        self.n_classes = <intp_t*>malloc(n_outputs * sizeof(intp_t))
        if self.n_classes is NULL:
            raise MemoryError("Failed to allocate n_classes array.")
        for i in range(n_outputs):
            self.n_classes[i] = n_classes[i]

        # Initialize tree state
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.nodes = NULL
        self.value = NULL

        # Allocate initial capacity
        self._resize_c(3)

    def __dealloc__(self):
        """Free all allocated memory."""
        free(self.nodes)
        free(self.value)
        free(self.n_classes)

    # ------------------------------------------------------------------
    # Construction from dict-based state (MaxFrame pipeline output)
    # ------------------------------------------------------------------

    @classmethod
    def from_state_dict(cls, state_dict):
        """Construct a Tree from MaxFrame's internal state dict.

        Parameters
        ----------
        state_dict : dict
            The tree state dict produced by the distributed training
            pipeline.  Expected keys: ``nodes`` (list of node dicts),
            ``n_features``, ``n_classes``, ``n_outputs``.

            Each node dict may contain: ``is_leaf``, ``left_child``,
            ``right_child``, ``feature``, ``threshold``, ``impurity``,
            ``n_node_samples``, ``weighted_n_node_samples``, ``value``.

        Returns
        -------
        Tree
            A populated Tree instance.
        """
        n_features = state_dict.get("n_features", 0)
        n_classes_val = np.asarray(
            state_dict.get("n_classes", 1), dtype=np.intp
        )
        if n_classes_val.ndim == 0:
            n_classes_val = n_classes_val.reshape(1)
        n_outputs = state_dict.get("n_outputs", 1)

        tree = cls(n_features, n_classes_val, n_outputs)
        tree._load_from_state(state_dict)
        return tree

    def _load_from_state(self, state_dict):
        """Populate C arrays from the dict-based node list.

        This is called internally by ``from_state_dict`` to fill the
        C-level Node structs and value arrays from the pipeline's dict
        representation.
        """
        nodes = state_dict.get("nodes", [])
        cdef intp_t n_nodes = len(nodes)
        if n_nodes == 0:
            return

        # Ensure capacity
        self._resize_c(n_nodes)

        cdef intp_t max_n_classes = self.max_n_classes
        cdef intp_t i
        cdef Node* node_ptr
        cdef bint is_leaf
        cdef intp_t left_child, right_child, feature
        cdef double threshold, impurity, weighted_n_node_samples
        cdef intp_t n_node_samples

        for i in range(n_nodes):
            node_dict = nodes[i]
            is_leaf = node_dict.get("is_leaf", True)
            node_ptr = &self.nodes[i]

            if is_leaf:
                node_ptr.left_child = TREE_LEAF
                node_ptr.right_child = TREE_LEAF
                node_ptr.feature = TREE_UNDEFINED
                node_ptr.threshold = TREE_UNDEFINED
                node_ptr.missing_go_to_left = 0
            else:
                node_ptr.left_child = <intp_t>node_dict.get("left_child", TREE_LEAF)
                node_ptr.right_child = <intp_t>node_dict.get("right_child", TREE_LEAF)
                node_ptr.feature = <intp_t>node_dict.get("feature", TREE_UNDEFINED)
                node_ptr.threshold = <double>node_dict.get("threshold", TREE_UNDEFINED)
                node_ptr.missing_go_to_left = 0

            node_ptr.impurity = <double>node_dict.get("impurity", 0.0)
            node_ptr.n_node_samples = <intp_t>node_dict.get("n_node_samples", 0)
            node_ptr.weighted_n_node_samples = <double>node_dict.get(
                "weighted_n_node_samples", 0.0
            )

            # Value: class counts (classification) or [mean] (regression)
            val = node_dict.get("value")
            if val is not None:
                val_arr = np.asarray(val, dtype=np.float64)
                if val_arr.ndim == 0:
                    val_arr = val_arr.reshape(1)
                # Flatten multi-output values (shape (n_outputs,
                # max_n_classes)) into the contiguous value buffer of
                # length value_stride = n_outputs * max_n_classes.
                val_flat = val_arr.ravel()
                if val_flat.shape[0] < self.value_stride:
                    padded = np.zeros(self.value_stride, dtype=np.float64)
                    padded[:val_arr.shape[0]] = val_arr
                    val_flat = padded
                for j in range(min(val_flat.shape[0], self.value_stride)):
                    self.value[i * self.value_stride + j] = val_flat[j]

        self.node_count = n_nodes
        self.capacity = n_nodes
        self.max_depth = self._compute_max_depth_from_state(nodes)

    cdef intp_t _compute_max_depth_from_state(self, list nodes):
        """Compute max depth from the dict-based node list via BFS."""
        if not nodes:
            return 0
        cdef dict depth = {0: 0}
        cdef intp_t max_depth = 0
        cdef object queue = deque([0])
        cdef intp_t node_id, left, right, d
        while queue:
            node_id = queue.popleft()
            node = nodes[node_id]
            d = depth[node_id]
            if d > max_depth:
                max_depth = d
            if not node.get("is_leaf", True):
                left = <intp_t>node.get("left_child", -1)
                right = <intp_t>node.get("right_child", -1)
                if left >= 0:
                    depth[left] = d + 1
                    queue.append(left)
                if right >= 0:
                    depth[right] = d + 1
                    queue.append(right)
        return max_depth

    # ------------------------------------------------------------------
    # sklearn-compatible array properties
    # ------------------------------------------------------------------

    @property
    def children_left(self):
        """ndarray of shape (node_count,): Left child indices. TREE_LEAF for leaves."""
        cdef cnp.ndarray nodes_arr = self._get_node_ndarray()
        return nodes_arr["left_child"].copy()

    @property
    def children_right(self):
        """ndarray of shape (node_count,): Right child indices. TREE_LEAF for leaves."""
        cdef cnp.ndarray nodes_arr = self._get_node_ndarray()
        return nodes_arr["right_child"].copy()

    @property
    def feature(self):
        """ndarray of shape (node_count,): Feature index per node. TREE_UNDEFINED for leaves."""
        cdef cnp.ndarray nodes_arr = self._get_node_ndarray()
        return nodes_arr["feature"].copy()

    @property
    def threshold(self):
        """ndarray of shape (node_count,): Threshold per node. TREE_UNDEFINED for leaves."""
        cdef cnp.ndarray nodes_arr = self._get_node_ndarray()
        return nodes_arr["threshold"].astype(np.float64).copy()

    @property
    def impurity(self):
        """ndarray of shape (node_count,): Impurity of each node."""
        cdef cnp.ndarray nodes_arr = self._get_node_ndarray()
        return nodes_arr["impurity"].copy()

    @property
    def n_node_samples(self):
        """ndarray of shape (node_count,): Number of training samples per node."""
        cdef cnp.ndarray nodes_arr = self._get_node_ndarray()
        return nodes_arr["n_node_samples"].copy()

    @property
    def weighted_n_node_samples(self):
        """ndarray of shape (node_count,): Weighted number of training samples per node."""
        cdef cnp.ndarray nodes_arr = self._get_node_ndarray()
        return nodes_arr["weighted_n_node_samples"].copy()

    @property
    def value(self):
        """ndarray of shape (node_count, n_outputs, max_n_classes): Prediction values."""
        return self._get_value_ndarray().copy()

    @property
    def n_leaves(self):
        """int: Number of leaf nodes."""
        cdef cnp.ndarray nodes_arr = self._get_node_ndarray()
        return int(np.sum(nodes_arr["left_child"] == TREE_LEAF))

    # ------------------------------------------------------------------
    # Internal C-level methods
    # ------------------------------------------------------------------

    cdef intp_t _resize_c(self, intp_t capacity) noexcept:
        """Resize the internal arrays to hold at least ``capacity`` nodes."""
        if capacity == 0 or capacity == self.capacity:
            if self.capacity == 0:
                capacity = 3
            else:
                capacity = 2 * self.capacity

        if capacity < self.node_count:
            capacity = self.node_count

        cdef size_t cap = <size_t>capacity
        # safe_realloc checks for integer overflow and NULL, raising
        # MemoryError on failure.  Catch it so the noexcept contract is
        # preserved and _resize_c returns -1 for the caller to handle.
        try:
            safe_realloc(<void**>&self.nodes, cap, sizeof(Node))
            safe_realloc(
                <void**>&self.value,
                cap * <size_t>self.value_stride,
                sizeof(double),
            )
        except MemoryError:
            return -1

        # Zero-initialize new memory
        if capacity > self.capacity:
            memset(
                <void*>(self.value + self.capacity * self.value_stride),
                0,
                (capacity - self.capacity) * self.value_stride * sizeof(double),
            )
            memset(
                <void*>(self.nodes + self.capacity),
                0,
                (capacity - self.capacity) * sizeof(Node),
            )

        self.capacity = capacity
        return capacity

    cdef intp_t _add_node(self, intp_t parent, bint is_left, bint is_leaf,
                          intp_t feature, double threshold, double impurity,
                          intp_t n_node_samples, double weighted_n_node_samples,
                          uint8_t missing_go_to_left) noexcept nogil:
        """Add a node to the tree (C-level, can be called without GIL)."""
        cdef intp_t node_id = self.node_count

        if node_id >= self.capacity:
            with gil:
                if self._resize_c(0) < 0:
                    return -1

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = TREE_LEAF
            node.right_child = TREE_LEAF
            node.feature = TREE_UNDEFINED
            node.threshold = TREE_UNDEFINED
            node.missing_go_to_left = 0
        else:
            node.feature = feature
            node.threshold = threshold
            node.missing_go_to_left = missing_go_to_left

        self.node_count += 1
        return node_id

    def add_node(self, intp_t parent, bint is_left, bint is_leaf,
                 intp_t feature, double threshold, double impurity,
                 intp_t n_node_samples, double weighted_n_node_samples,
                 uint8_t missing_go_to_left=0):
        """Add a node to the tree (Python-accessible wrapper).

        Parameters
        ----------
        parent : int
            Index of the parent node, or -2 for the root.
        is_left : bool
            True if this is the left child of the parent.
        is_leaf : bool
            True if this is a leaf node.
        feature : int
            Feature index for the split.
        threshold : double
            Threshold value for the split.
        impurity : double
            Impurity of the node.
        n_node_samples : int
            Number of samples at this node.
        weighted_n_node_samples : double
            Weighted number of samples at this node.
        missing_go_to_left : uint8_t, default=0
            Whether missing values go to the left child.

        Returns
        -------
        int
            The index of the newly added node.
        """
        return self._add_node(
            parent, is_left, is_leaf, feature, threshold, impurity,
            n_node_samples, weighted_n_node_samples, missing_go_to_left,
        )

    def _shrink_to_fit(self):
        """Shrink the internal arrays to exactly fit the current node count."""
        self._resize_c(self.node_count)

    cdef cnp.ndarray _apply_dense(self, cnp.ndarray X):
        """Find the leaf index for each sample in X (dense input)."""
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                f"was trained with {self.n_features} features."
            )
        cdef intp_t n_samples = X.shape[0]
        cdef cnp.ndarray out = np.zeros(n_samples, dtype=np.intp)
        cdef cnp.ndarray X_contig = np.ascontiguousarray(X, dtype=np.float64)
        cdef double* X_data = <double*>X_contig.data
        cdef intp_t* out_data = <intp_t*>out.data
        cdef intp_t X_n_features = self.n_features
        cdef intp_t i
        cdef Node* node
        cdef double feature_value

        for i in range(n_samples):
            node = &self.nodes[0]
            while node.left_child != TREE_LEAF:
                feature_value = X_data[i * X_n_features + node.feature]
                if isnan(feature_value):
                    if node.missing_go_to_left:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]
                elif feature_value <= node.threshold:
                    node = &self.nodes[node.left_child]
                else:
                    node = &self.nodes[node.right_child]
            out_data[i] = <intp_t>(node - self.nodes)

        return out

    cpdef cnp.ndarray apply(self, object X):
        """Return the leaf index for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            The leaf index for each sample.
        """
        cdef cnp.ndarray X_ndarray = np.asarray(X, dtype=np.float64)
        if X_ndarray.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {X_ndarray.ndim}D array instead."
            )
        return self._apply_dense(X_ndarray)

    cpdef cnp.ndarray predict(self, object X):
        """Predict target for X.

        For classification, returns class probabilities for each sample.
        For regression, returns the predicted value for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples, n_outputs * max_n_classes)
            The predicted values.
        """
        cdef cnp.ndarray X_ndarray = np.asarray(X, dtype=np.float64)
        if X_ndarray.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {X_ndarray.ndim}D array instead."
            )
        out = self._get_value_ndarray().take(
            self._apply_dense(X_ndarray), axis=0, mode='clip'
        )
        if self.n_outputs == 1:
            out = out.reshape(X_ndarray.shape[0], self.max_n_classes)
        return out

    cpdef object decision_path(self, object X):
        """Return the decision path for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        sparse matrix of shape (n_samples, n_nodes)
            CSR sparse matrix where entry [i, j] indicates that
            sample i passes through node j.
        """
        from scipy.sparse import csr_matrix

        cdef cnp.ndarray X_ndarray = np.asarray(X, dtype=np.float64)
        if X_ndarray.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {X_ndarray.ndim}D array instead."
            )

        cdef intp_t n_samples = X_ndarray.shape[0]
        cdef cnp.ndarray X_contig = np.ascontiguousarray(X_ndarray, dtype=np.float64)
        cdef double* X_data = <double*>X_contig.data
        cdef intp_t X_n_features = self.n_features

        cdef list indptr_list = [0]
        cdef list indices_list = []

        cdef intp_t i
        cdef Node* node
        cdef intp_t node_idx
        cdef double feature_value

        for i in range(n_samples):
            node = &self.nodes[0]
            node_idx = 0
            indices_list.append(node_idx)

            while node.left_child != TREE_LEAF:
                feature_value = X_data[i * X_n_features + node.feature]
                if isnan(feature_value):
                    if node.missing_go_to_left:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]
                elif feature_value <= node.threshold:
                    node = &self.nodes[node.left_child]
                else:
                    node = &self.nodes[node.right_child]
                node_idx = <intp_t>(node - self.nodes)
                indices_list.append(node_idx)

            indptr_list.append(len(indices_list))

        cdef cnp.ndarray data = np.ones(len(indices_list), dtype=np.float64)
        cdef cnp.ndarray indices = np.array(indices_list, dtype=np.intp)
        cdef cnp.ndarray indptr = np.array(indptr_list, dtype=np.intp)

        return csr_matrix(
            (data, indices, indptr),
            shape=(n_samples, self.node_count),
        )

    cpdef compute_feature_importances(self, normalize=True):
        """Compute feature importances based on impurity reduction.

        The importance of a feature is computed as the total reduction
        of the criterion brought by that feature, weighted by the number
        of samples at each node. This is also known as Gini importance.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to normalize the importances to sum to 1.

        Returns
        -------
        ndarray of shape (n_features,)
            The feature importances.
        """
        cdef cnp.ndarray importances = np.zeros(self.n_features, dtype=np.float64)
        cdef double[:] importances_view = importances

        cdef Node* node = self.nodes
        cdef Node* end_node = self.nodes + self.node_count
        cdef Node* left
        cdef Node* right
        cdef double normalizer = 0.0
        cdef int i

        while node < end_node:
            if node.left_child != TREE_LEAF:
                left = &self.nodes[node.left_child]
                right = &self.nodes[node.right_child]

                importances_view[node.feature] += (
                    node.weighted_n_node_samples * node.impurity
                    - left.weighted_n_node_samples * left.impurity
                    - right.weighted_n_node_samples * right.impurity
                )
            node += 1

        # Normalize by total weight at root (guard against division by zero
        # when weighted_n_node_samples is not populated by the training pipeline)
        if self.node_count > 0 and self.nodes[0].weighted_n_node_samples > 0.0:
            for i in range(self.n_features):
                importances_view[i] /= self.nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)
            if normalizer > 0.0:
                for i in range(self.n_features):
                    importances_view[i] /= normalizer

        return importances

    cdef cnp.ndarray _get_value_ndarray(self):
        """Return the value array as a numpy ndarray."""
        if self.node_count == 0:
            return np.empty((0, self.n_outputs, self.max_n_classes), dtype=np.float64)
        cdef cnp.ndarray value_array = np.asarray(
            <double[:self.node_count, :self.n_outputs, :self.max_n_classes]>
            self.value
        )
        return value_array

    cdef cnp.ndarray _get_node_ndarray(self):
        """Return the nodes array as a numpy structured ndarray."""
        # The C struct has padding after missing_go_to_left (uint8_t)
        # to align to the struct's alignment. The padding size depends on
        # the platform (e.g. 7 bytes on 64-bit, 3 bytes on 32-bit), so we
        # compute it dynamically from sizeof(Node).
        cdef Py_ssize_t field_bytes = (
            3 * sizeof(intp_t)      # left_child, right_child, feature
            + 2 * sizeof(double)    # threshold, impurity
            + sizeof(intp_t)        # n_node_samples
            + sizeof(double)        # weighted_n_node_samples
            + sizeof(uint8_t)       # missing_go_to_left
        )
        cdef Py_ssize_t padding = sizeof(Node) - field_bytes
        dtype = np.dtype([
            ("left_child", np.intp),
            ("right_child", np.intp),
            ("feature", np.intp),
            ("threshold", np.float64),
            ("impurity", np.float64),
            ("n_node_samples", np.intp),
            ("weighted_n_node_samples", np.float64),
            ("missing_go_to_left", np.uint8),
            ("_padding", np.uint8, padding),
        ])
        if self.node_count == 0:
            # Return an empty array with the full Node dtype so that all
            # sklearn-compatible properties (children_left/right, feature,
            # threshold, impurity, n_node_samples, weighted_n_node_samples)
            # work on empty trees instead of raising a missing-field error.
            return np.empty(0, dtype=dtype)
        cdef cnp.ndarray nodes_array = np.frombuffer(
            <char[:self.node_count * sizeof(Node)]>(
                <char*>self.nodes
            ),
            dtype=dtype,
        )
        return nodes_array

    def __getstate__(self):
        """Serialize the tree for distributed transfer.

        Converts the C arrays to numpy arrays for pickle/cloudpickle
        serialization, enabling MaxFrame's ObjectData pattern to
        transfer trained trees to prediction workers.

        Returns
        -------
        dict
            Serialized tree state.
        """
        return {
            "n_features": self.n_features,
            "n_outputs": self.n_outputs,
            "max_n_classes": self.max_n_classes,
            "max_depth": self.max_depth,
            "node_count": self.node_count,
            "capacity": self.capacity,
            "n_classes": np.asarray(
                <intp_t[:self.n_outputs]>self.n_classes
            ).copy(),
            "nodes": self._get_node_ndarray().copy(),
            "values": self._get_value_ndarray().copy(),
        }

    def __setstate__(self, state):
        """Deserialize the tree from a state dict.

        Supports two formats:

        1. **Cython serialization format** (from ``__getstate__``): contains
           ``nodes`` as a numpy structured ndarray and ``values`` as a numpy
           ndarray.
        2. **Dict-based format** (from MaxFrame pipeline): contains ``nodes``
           as a list of node dicts with keys like ``is_leaf``, ``left_child``,
           ``threshold``, ``value``, etc.

        Parameters
        ----------
        state : dict
            Serialized tree state from ``__getstate__`` or the training
            pipeline.
        """
        cdef int i, k
        cdef cnp.ndarray n_classes_np, nodes_np, values_np
        cdef intp_t n_nodes

        # Detect format: dict-based nodes (list) vs Cython (structured ndarray)
        nodes_data = state.get("nodes")
        is_dict_format = isinstance(nodes_data, list)

        if is_dict_format:
            # Dict-based format from MaxFrame training pipeline.
            # Re-initialise from scratch using from_state_dict logic.
            n_features = state.get("n_features", 0)
            n_classes_val = np.asarray(
                state.get("n_classes", 1), dtype=np.intp
            )
            if n_classes_val.ndim == 0:
                n_classes_val = n_classes_val.reshape(1)
            n_outputs = state.get("n_outputs", 1)

            self.n_features = n_features
            self.n_outputs = n_outputs
            self.max_n_classes = <intp_t>np.max(n_classes_val)
            self.value_stride = n_outputs * self.max_n_classes

            # Free existing memory
            free(self.n_classes)
            self.n_classes = NULL
            free(self.nodes)
            self.nodes = NULL
            free(self.value)
            self.value = NULL

            # Reset state
            self.max_depth = 0
            self.node_count = 0
            self.capacity = 0

            # Allocate n_classes C array
            self.n_classes = <intp_t*>malloc(n_outputs * sizeof(intp_t))
            if self.n_classes is NULL:
                raise MemoryError("Failed to allocate n_classes array.")
            for k in range(n_outputs):
                self.n_classes[k] = n_classes_val[k]

            # Allocate initial capacity
            self._resize_c(3)

            # Populate from dict-based node list
            self._load_from_state(state)
            return

        # Cython serialization format (structured ndarray)
        self.n_features = state["n_features"]
        self.n_outputs = state["n_outputs"]
        self.max_n_classes = state["max_n_classes"]
        self.max_depth = state["max_depth"]
        self.node_count = state["node_count"]
        self.capacity = state["capacity"]
        self.value_stride = self.n_outputs * self.max_n_classes

        # Free existing memory and null out pointers
        free(self.n_classes)
        self.n_classes = NULL
        free(self.nodes)
        self.nodes = NULL
        free(self.value)
        self.value = NULL

        # Reconstruct n_classes C array
        n_classes_np = np.asarray(state["n_classes"], dtype=np.intp)
        self.n_classes = <intp_t*>malloc(self.n_outputs * sizeof(intp_t))
        if self.n_classes is NULL:
            raise MemoryError("Failed to allocate n_classes array.")
        for i in range(self.n_outputs):
            self.n_classes[i] = n_classes_np[i]

        # Reconstruct nodes C array
        nodes_np = np.asarray(state["nodes"])
        n_nodes = nodes_np.shape[0]
        cdef size_t nodes_alloc = <size_t>self.capacity * sizeof(Node)
        if nodes_alloc / sizeof(Node) != <size_t>self.capacity:
            raise MemoryError("Overflow detected in nodes allocation.")
        self.nodes = <Node*>malloc(nodes_alloc)
        if self.nodes is NULL:
            raise MemoryError("Failed to allocate nodes array.")
        memcpy(
            <void*>self.nodes,
            <void*><char*>nodes_np.data,
            n_nodes * sizeof(Node),
        )

        # Reconstruct value C array
        values_np = np.asarray(state["values"], dtype=np.float64)
        cdef size_t value_items = <size_t>self.capacity * <size_t>self.value_stride
        cdef size_t value_alloc = value_items * sizeof(double)
        if value_alloc / sizeof(double) != value_items:
            raise MemoryError("Overflow detected in value allocation.")
        self.value = <double*>malloc(value_alloc)
        if self.value is NULL:
            raise MemoryError("Failed to allocate value array.")
        memcpy(
            <void*>self.value,
            <void*><char*>values_np.data,
            values_np.size * sizeof(double),
        )

    def __reduce__(self):
        """Support for pickling."""
        return (Tree, (self.n_features, np.asarray(
            <intp_t[:self.n_outputs]>self.n_classes
        ).copy(), self.n_outputs), self.__getstate__())

    @property
    def n_classes_array(self):
        """Return n_classes as a numpy array."""
        return np.asarray(
            <intp_t[:self.n_outputs]>self.n_classes
        ).copy()

    def __repr__(self):
        return (
            f"Tree(n_features={self.n_features}, "
            f"n_outputs={self.n_outputs}, "
            f"n_nodes={self.node_count}, "
            f"max_depth={self.max_depth})"
        )
