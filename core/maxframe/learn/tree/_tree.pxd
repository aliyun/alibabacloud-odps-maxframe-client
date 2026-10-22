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

from libc.stdint cimport uint8_t

cimport numpy as cnp

cnp.import_array()

ctypedef Py_ssize_t intp_t

# Sentinel values
cdef intp_t TREE_LEAF
cdef intp_t TREE_UNDEFINED

# Node struct
cdef struct Node:
    intp_t left_child
    intp_t right_child
    intp_t feature
    double threshold
    double impurity
    intp_t n_node_samples
    double weighted_n_node_samples
    uint8_t missing_go_to_left


cdef class Tree:
    cdef public intp_t n_features
    cdef intp_t* n_classes
    cdef public intp_t n_outputs
    cdef public intp_t max_n_classes
    cdef public intp_t max_depth
    cdef public intp_t node_count
    cdef public intp_t capacity
    cdef Node* nodes
    cdef double* value
    cdef intp_t value_stride

    cdef intp_t _add_node(self, intp_t parent, bint is_left, bint is_leaf,
                          intp_t feature, double threshold, double impurity,
                          intp_t n_node_samples, double weighted_n_node_samples,
                          uint8_t missing_go_to_left) noexcept nogil
    cdef intp_t _resize_c(self, intp_t capacity) noexcept
    cdef cnp.ndarray _apply_dense(self, cnp.ndarray X)
    cdef cnp.ndarray _get_value_ndarray(self)
    cdef cnp.ndarray _get_node_ndarray(self)
    cdef intp_t _compute_max_depth_from_state(self, list nodes)
    cpdef cnp.ndarray predict(self, object X)
    cpdef cnp.ndarray apply(self, object X)
    cpdef object decision_path(self, object X)
    cpdef compute_feature_importances(self, normalize=*)
