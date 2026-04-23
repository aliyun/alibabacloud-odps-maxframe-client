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

from ... import utils


def test_find_objects():
    # Test with types filter
    data = [1, "string", [2, 3], {"key": "value"}, 4.5]
    strings = utils.find_objects(data, str)
    assert strings == ["string", "value"]

    # Test with multiple types
    numbers = utils.find_objects(data, (int, float))
    assert set(numbers) == {1, 2, 3, 4.5}  # find_objects finds nested numbers too

    # Test with checker function
    def is_even(x):
        return isinstance(x, int) and x % 2 == 0

    even_numbers = utils.find_objects(data, checker=is_even)
    assert even_numbers == [2]  # Only 2 is even

    # Test with nested structure
    nested = {
        "level1": [{"level2": "found"}, {"level2": [1, 2, 3]}],
        "another": "not_found",
    }
    strings = utils.find_objects(nested, str)
    assert set(strings) == {"found", "not_found"}

    # Test empty structure
    assert utils.find_objects([], str) == []
    assert utils.find_objects({}, str) == []


def test_replace_objects():
    # Test simple replacement
    data = [1, 2, 3, 4]
    mapping = {1: "one", 3: "three"}
    result = utils.replace_objects(data, mapping)
    assert result == ["one", 2, "three", 4]

    # Test with dict
    data = {"a": 1, "b": 2, "c": 3}
    mapping = {1: "one", 3: "three"}
    result = utils.replace_objects(data, mapping)
    assert result == {"a": "one", "b": 2, "c": "three"}

    # Test with nested structure
    data = [1, [2, 3], {"a": 4, "b": [5, 6]}]
    mapping = {1: "one", 3: "three", 5: "five"}
    result = utils.replace_objects(data, mapping)
    assert result == ["one", [2, "three"], {"a": 4, "b": ["five", 6]}]

    # Test with empty mapping
    data = [1, 2, 3]
    result = utils.replace_objects(data, {})
    assert result == [1, 2, 3]

    # Test with tuple
    data = (1, 2, 3)
    mapping = {1: "one", 3: "three"}
    result = utils.replace_objects(data, mapping)
    assert result == ("one", 2, "three")

    # Test with set
    data = {1, 2, 3}
    mapping = {1: "one", 3: "three"}
    result = utils.replace_objects(data, mapping)
    # Note: order doesn't matter for sets
    assert set(result) == {2, "one", "three"}


def test_is_empty():
    import numpy as np
    import pandas as pd

    # Test with pandas objects
    empty_df = pd.DataFrame()
    non_empty_df = pd.DataFrame({"a": [1, 2, 3]})
    assert utils.is_empty(empty_df) is True
    assert utils.is_empty(non_empty_df) is False

    empty_series = pd.Series([], dtype=object)
    non_empty_series = pd.Series([1, 2, 3])
    assert utils.is_empty(empty_series) is True
    assert utils.is_empty(non_empty_series) is False

    empty_index = pd.Index([])
    non_empty_index = pd.Index([1, 2, 3])
    assert utils.is_empty(empty_index) is True
    assert utils.is_empty(non_empty_index) is False

    # Test with basic Python types
    assert utils.is_empty([]) is True
    assert utils.is_empty([1, 2, 3]) is False
    assert utils.is_empty({}) is True
    assert utils.is_empty({"a": 1}) is False
    assert utils.is_empty("") is True
    assert utils.is_empty("hello") is False
    assert utils.is_empty(0) is True
    assert utils.is_empty(1) is False
    assert utils.is_empty(None) is True

    # Test with numpy arrays
    empty_array = np.array([])
    non_empty_array = np.array([1, 2, 3])
    assert utils.is_empty(empty_array) is True
    assert utils.is_empty(non_empty_array) is False


def test_flatten():
    # Test basic flattening
    assert utils.flatten([[0, 1], [2, 3]]) == [0, 1, 2, 3]
    assert utils.flatten([1, [2, 3], 4]) == [1, 2, 3, 4]

    # Test nested flattening
    assert utils.flatten([[0, 1], [[3], [4, 5]]]) == [0, 1, 3, 4, 5]
    assert utils.flatten([1, [2, [3, [4, 5]]]]) == [1, 2, 3, 4, 5]

    # Test with tuples
    assert utils.flatten([(0, 1), (2, 3)]) == [0, 1, 2, 3]
    assert utils.flatten([1, (2, 3), 4]) == [1, 2, 3, 4]

    # Test mixed lists and tuples
    assert utils.flatten([1, [2, (3, 4)], 5]) == [1, 2, 3, 4, 5]

    # Test empty structures
    assert utils.flatten([]) == []
    assert utils.flatten([[], []]) == []
    assert utils.flatten([[], [1, []], []]) == [1]

    # Test single level
    assert utils.flatten([1, 2, 3]) == [1, 2, 3]

    # Test deep nesting
    deep = [[[[[1]]]], 2, [[[3, 4]], 5]]
    assert utils.flatten(deep) == [1, 2, 3, 4, 5]


def test_stack_back():
    # Test basic stacking
    raw = [[0, 1], [2, [3, 4]]]
    flattened = utils.flatten(raw)
    assert flattened == [0, 1, 2, 3, 4]

    # Modify flattened values
    modified = [x + 1 for x in flattened]
    result = utils.stack_back(modified, raw)
    assert result == [[1, 2], [3, [4, 5]]]

    # Test with different structure
    raw = [1, [2, 3], [[4], 5]]
    flattened = utils.flatten(raw)
    assert flattened == [1, 2, 3, 4, 5]

    modified = [x * 2 for x in flattened]
    result = utils.stack_back(modified, raw)
    assert result == [2, [4, 6], [[8], 10]]

    # Test with tuples inside list (root is list)
    raw = [(0, 1), (2, (3, 4))]
    flattened = utils.flatten(raw)
    assert flattened == [0, 1, 2, 3, 4]

    modified = [x + 10 for x in flattened]
    result = utils.stack_back(modified, raw)
    # Root is list, nested tuples preserved
    assert result == [(10, 11), (12, (13, 14))]

    # Test empty structure
    raw = []
    flattened = []
    result = utils.stack_back(flattened, raw)
    assert result == []

    # Test single level
    raw = [1, 2, 3]
    flattened = utils.flatten(raw)
    modified = [x + 1 for x in flattened]
    result = utils.stack_back(modified, raw)
    assert result == [2, 3, 4]

    # Test complex nested structure
    raw = [[1, [2, 3]], [[4], 5], 6]
    flattened = utils.flatten(raw)
    assert flattened == [1, 2, 3, 4, 5, 6]

    modified = [f"item_{x}" for x in flattened]
    result = utils.stack_back(modified, raw)
    assert result == [
        ["item_1", ["item_2", "item_3"]],
        [["item_4"], "item_5"],
        "item_6",
    ]


def test_integration_flatten_stack_back():
    # Test that flatten and stack_back are inverse operations
    test_cases = [
        [1, 2, 3],
        [[1, 2], [3, 4]],
        [1, [2, [3, [4, 5]]]],
        [(1, 2), [3, (4, 5)]],
        [],
        [[], [[], []]],
        [{"a": 1}, [{"b": 2}, 3]],
    ]

    for raw in test_cases:
        # Only test with list/tuple structures (skip dicts for now)
        if not any(
            isinstance(x, dict)
            for x in utils.flatten(raw)
            if isinstance(x, (list, tuple))
        ):
            flattened = utils.flatten(raw)
            # Apply some transformation
            transformed = [
                x * 2 if isinstance(x, (int, float)) else x for x in flattened
            ]
            # Stack back
            result = utils.stack_back(transformed, raw)
            # The structure should be preserved
            assert len(result) == len(raw)
            # Flatten result to verify transformation was applied
            result_flattened = utils.flatten(result)
            assert result_flattened == transformed


def test_lru_cache():
    # Test 1: Basic get/set operations
    cache = utils.LRUDict(maxsize=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert cache["a"] == 1
    assert cache["b"] == 2
    assert cache["c"] == 3
    assert len(cache) == 3
    # Test 2: Evict least recently used element when exceeding maxsize
    cache["d"] = 4  # 'a' should be evicted (a was accessed earliest among a, b, c)
    assert len(cache) == 3
    assert "a" not in cache  # 'a' is evicted
    assert "b" in cache
    assert "c" in cache
    assert "d" in cache
    # Test 3: Accessed element moves to end and won't be evicted first
    cache2 = utils.LRUDict(maxsize=3)
    cache2["x"] = 10
    cache2["y"] = 20
    cache2["z"] = 30
    _ = cache2["x"]  # Access 'x', making it most recently used
    cache2["w"] = 40  # 'y' should be evicted (least recently used)
    assert "x" in cache2
    assert "y" not in cache2
    assert "z" in cache2
    assert "w" in cache2
    # Test 4: Updating existing key doesn't increase length
    cache3 = utils.LRUDict(maxsize=2)
    cache3["a"] = 1
    cache3["b"] = 2
    cache3["a"] = 100  # Update 'a'
    assert len(cache3) == 2
    assert cache3["a"] == 100
    # Test 5: Updated key moves to end (most recently used)
    cache3["c"] = 3  # 'b' should be evicted since 'a' was just updated
    assert "a" in cache3
    assert "b" not in cache3
    assert "c" in cache3
    # Test 6: Initialize from dict using update
    cache4 = utils.LRUDict(maxsize=5)
    data = {"k1": "v1", "k2": "v2", "k3": "v3"}
    cache4.update(data)
    assert len(cache4) == 3
    assert cache4["k1"] == "v1"
    assert cache4["k2"] == "v2"
    assert cache4["k3"] == "v3"
    # Test 7: Edge case with maxsize=1
    cache5 = utils.LRUDict(maxsize=1)
    cache5["only"] = "one"
    assert cache5["only"] == "one"
    cache5["new"] = "value"
    assert "only" not in cache5
    assert cache5["new"] == "value"
    assert len(cache5) == 1
    # Test 8: Accessing non-existent key raises KeyError
    cache6 = utils.LRUDict(maxsize=3)
    cache6["a"] = 1
    try:
        _ = cache6["nonexistent"]
        assert False, "Expected KeyError"
    except KeyError:
        pass  # Expected
