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

from ....dataframe import core  # noqa: F401  # pylint: disable=unused-variable
from ... import OutputType
from .. import Operator, TileableOperatorMixin


class MyOperator(Operator, TileableOperatorMixin):
    @classmethod
    def execute(cls, ctx, op):
        return 1

    @classmethod
    def estimate_size(cls, ctx, op):
        return 1


class MyOperator2(MyOperator):
    @classmethod
    def execute(cls, ctx, op):
        raise NotImplementedError

    @classmethod
    def estimate_size(cls, ctx, op):
        raise NotImplementedError


class _OperatorMixin(TileableOperatorMixin):
    pass


class MyOperator3(Operator, _OperatorMixin):
    pass


class MyOperator4(Operator, _OperatorMixin):
    pass


class MyOperator5(MyOperator4):
    pass


def test_unknown_dtypes():
    op = MyOperator(_output_types=[OutputType.dataframe])
    df = op.new_tileable(None, dtypes=None)
    op2 = MyOperator(_output_types=[OutputType.scalar])
    with pytest.raises(ValueError) as exc_info:
        op2.new_tileable([df])
    assert "executed first" in exc_info.value.args[0]
