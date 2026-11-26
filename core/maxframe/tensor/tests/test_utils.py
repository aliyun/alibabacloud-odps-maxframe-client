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

import pytest

from maxframe.tensor.utils import slice_split

EMPTY_CASES = [
    (slice(0, 0), [5]),
    (slice(2, 2), [2, 2, 1]),
    (slice(5, 5), [5]),
    (slice(5, None), [5]),
    (slice(3, 1), [2, 2, 1]),
]


@pytest.mark.parametrize(
    "index, sizes",
    EMPTY_CASES,
    ids=["0:0", "2:2", "5:5", "5:", "3:1"],
)
def test_slice_split_empty_slice_default(index, sizes):
    """Without the flag an empty slice still lands on chunk 0."""
    assert slice_split(index, sizes) == {0: slice(0, 0, 1)}


@pytest.mark.parametrize(
    "index, sizes, expected",
    [
        (slice(0, 0), [2, 2, 1], {0: slice(0, 0, 1)}),
        (slice(2, 2), [2, 2, 1], {0: slice(2, 2, 1)}),
        (slice(5, 5), [5], {0: slice(5, 5, 1)}),
        (slice(5, None), [5], {0: slice(5, 5, 1)}),
        (slice(5, 5), [2, 2, 1], {2: slice(1, 1, 1)}),
        (slice(3, 1), [2, 2, 1], {1: slice(1, 1, 1)}),
    ],
)
def test_slice_split_empty_slice_keeps_position(index, sizes, expected):
    assert slice_split(index, sizes, keep_empty_position=True) == expected


def test_slice_split_non_empty_slice_unchanged():
    expected = {0: slice(2, 5, 1), 1: slice(0, 3, 1)}
    assert slice_split(slice(2, 8), [5, 5, 5]) == expected
    assert slice_split(slice(2, 8), [5, 5, 5], keep_empty_position=True) == expected
