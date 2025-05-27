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

import base64
from typing import List, Tuple

import pytest

from ...lib import wrapped_pickle
from ...serialization.core import PickleContainer
from ..core import UserCodeMixin


@pytest.mark.parametrize(
    "input_obj, expected_output",
    [
        (None, "None"),
        (10, "10"),
        (3.14, "3.14"),
        (True, "True"),
        (False, "False"),
        (b"hello", "base64.b64decode(b'aGVsbG8=')"),
        ("hello", "'hello'"),
        ([1, 2, 3], "[1, 2, 3]"),
        ({"a": 1, "b": 2}, "{'a': 1, 'b': 2}"),
        ((1, 2, 3), "(1, 2, 3)"),
        ((1,), "(1,)"),
        ((), "()"),
        ({1, 2, 3}, "{1, 2, 3}"),
        (set(), "set()"),
    ],
)
def test_obj_to_python_expr(input_obj, expected_output):
    assert UserCodeMixin.obj_to_python_expr(input_obj) == expected_output


def test_obj_to_python_expr_custom_object():
    class CustomClass:
        def __init__(self, a: int, b: List[int], c: Tuple[int, int]):
            self.a = a
            self.b = b
            self.c = c

    custom_obj = CustomClass(1, [2, 3], (4, 5))
    pickle_data = wrapped_pickle.dumps(custom_obj)
    pickle_str = base64.b64encode(pickle_data)
    custom_obj_pickle_container = PickleContainer([pickle_data])

    # with class obj will not support currently
    with pytest.raises(ValueError):
        UserCodeMixin.obj_to_python_expr(custom_obj)

    assert (
        UserCodeMixin.obj_to_python_expr(custom_obj_pickle_container)
        == f"cloudpickle.loads(base64.b64decode({pickle_str}), buffers=[])"
    )
