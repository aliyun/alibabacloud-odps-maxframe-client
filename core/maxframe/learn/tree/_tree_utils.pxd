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

cdef inline void safe_realloc(void** p_ptr, size_t n_items, size_t item_size) except *:
    """Reallocate memory with overflow checking.

    Parameters
    ----------
    p_ptr : void**
        Pointer to the memory block to reallocate.
    n_items : size_t
        Number of items to allocate.
    item_size : size_t
        Size of each item in bytes.

    Raises
    ------
    MemoryError
        If the allocation fails or would overflow.
    """
    cdef size_t n_bytes = n_items * item_size
    if n_bytes / item_size != n_items:
        raise MemoryError(
            "Failed to allocate memory: overflow detected. "
            f"Requested {n_items} items of size {item_size}."
        )

    cdef void* new_ptr = realloc(p_ptr[0], n_bytes)
    if new_ptr is NULL:
        raise MemoryError(
            f"Failed to allocate {n_bytes} bytes."
        )
    p_ptr[0] = new_ptr


cdef extern from "stdlib.h":
    void *realloc(void *ptr, size_t size) noexcept
