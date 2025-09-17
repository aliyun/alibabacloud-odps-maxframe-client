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

import pytest

from ..validators import (
    is_less_than_or_equal_to,
    is_positive_integer,
    simple_yaml_str_validator,
)


@pytest.mark.parametrize("value", ["a", "http://127.0.0.1:1234", "a-b#", "ab_", "123"])
def test_simple_yaml_str_validator_valid(value):
    assert simple_yaml_str_validator(value)


@pytest.mark.parametrize("value", ['a"', "'hacked'", "ab\n", "abc\\"])
def test_simple_yaml_str_validator_invalid(value):
    assert not simple_yaml_str_validator(value)


@pytest.mark.parametrize(
    "value,valid", [("a", False), (1, True), (0, False), (-1, False)]
)
def test_is_positive_integer_validator(value, valid):
    assert is_positive_integer(value) is valid


@pytest.mark.parametrize(
    "value,upper_bound,valid",
    [(3, 5, True), (5, 5, True), (6, 5, False), (None, None, False), (None, 5, False)],
)
def test_is_less_than_or_equal_to_validator(value, upper_bound, valid):
    assert is_less_than_or_equal_to(upper_bound)(value) is valid
