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

from typing import Callable
from urllib.parse import urlparse

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


class Validator:
    def __init__(self, func: ValidatorType):
        self._func = func

    def __call__(self, arg) -> bool:
        return self._func(arg)

    def __or__(self, other):
        return OrValidator(self, other)


class OrValidator(Validator):
    def __init__(self, lhs: Validator, rhs: Validator):
        super().__init__(lambda x: lhs(x) or rhs(x))


is_null = Validator(lambda x: x is None)
is_bool = Validator(lambda x: isinstance(x, bool))
is_float = Validator(lambda x: isinstance(x, float))
is_integer = Validator(lambda x: isinstance(x, int))
is_numeric = Validator(lambda x: isinstance(x, (int, float)))
is_string = Validator(lambda x: isinstance(x, str))
is_dict = Validator(lambda x: isinstance(x, dict))
is_positive_integer = Validator(lambda x: is_integer(x) and x > 0)
is_non_negative_integer = Validator(lambda x: is_integer(x) and x >= 0)


def is_in(vals):
    return Validator(vals.__contains__)


def _is_valid_cache_path(path: str) -> bool:
    """
    path should look like oss://oss_endpoint/oss_bucket/path
    """
    parsed_url = urlparse(path)
    return (
        parsed_url.scheme == "oss"
        and parsed_url.netloc
        and parsed_url.path
        and "/" in parsed_url.path
    )


is_valid_cache_path = Validator(_is_valid_cache_path)


_invalid_char_in_yaml_str = {'"', "'", "\n", "\\"}


def simple_yaml_str_validator(name: str) -> bool:
    chars = set(name)
    return len(_invalid_char_in_yaml_str & chars) == 0
