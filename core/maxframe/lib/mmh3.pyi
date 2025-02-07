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

from typing import Tuple

def hash(key, seed=0, signed=True) -> int:
    """
    Return a 32 bit integer.
    """

def hash_from_buffer(key, seed=0, signed=True) -> int:
    """
    Return a 32 bit integer. Designed for large memory-views such as numpy arrays.
    """

def hash64(key, seed=0, x64arch=True, signed=True) -> Tuple[int, int]:
    """
    Return a tuple of two 64 bit integers for a string. Optimized for
    the x64 bit architecture when x64arch=True, otherwise for x86.
    """

def hash128(key, seed=0, x64arch=True, signed=False) -> int:
    """
    Return a 128 bit long integer. Optimized for the x64 bit architecture
    when x64arch=True, otherwise for x86.
    """

def hash_bytes(key, seed=0, x64arch=True) -> bytes:
    """
    Return a 128 bit hash value as bytes for a string. Optimized for the
    x64 bit architecture when x64arch=True, otherwise for the x86.
    """
