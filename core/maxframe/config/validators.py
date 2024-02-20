# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

from typing import Callable

ValidatorType = Callable[..., bool]


def any_validator(*validators: ValidatorType):
    def validate(x):
        return any(validator(x) for validator in validators)

    return validate


def all_validator(*validators: ValidatorType):
    def validate(x):
        return all(validator(x) for validator in validators)

    validate.validators = validators
    return validate


is_null = lambda x: x is None
is_bool = lambda x: isinstance(x, bool)
is_float = lambda x: isinstance(x, float)
is_integer = lambda x: isinstance(x, int)
is_numeric = lambda x: isinstance(x, (int, float))
is_string = lambda x: isinstance(x, str)
is_dict = lambda x: isinstance(x, dict)
is_positive_integer = lambda x: is_integer(x) and x > 0


def is_in(vals):
    def validate(x):
        return x in vals

    return validate


_invalid_char_in_yaml_str = {'"', "'", "\n", "\\"}


def simple_yaml_str_validator(name: str) -> bool:
    chars = set(name)
    return len(_invalid_char_in_yaml_str & chars) == 0
