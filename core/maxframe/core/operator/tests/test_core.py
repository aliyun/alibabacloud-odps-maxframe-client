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

from maxframe.core import OutputType
from maxframe.core.operator import Operator, TileableOperatorMixin
from maxframe.core.operator.base import SchedulingHint
from maxframe.dataframe import core  # noqa: F401  # pylint: disable=unused-variable
from maxframe.udf import FsMountOptions


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


def test_scheduling_hint_copy_detaches_containers():
    hint = SchedulingHint(
        pure_depends=[True],
        expect_resources={"cpu": 1, "memory": [2, 3]},
        fs_mount=[FsMountOptions(path="/tmp/a", mount_path="/mnt/a")],
        image_options={"name": "image"},
        network_configs={"public_network_whitelist": ["10.0.0.1"]},
    )

    copied = hint.copy()
    assert copied.expect_resources == {"cpu": 1, "memory": [2, 3]}

    copied._pure_depends.append(False)
    copied.expect_resources["cpu"] = 4
    copied.expect_resources["memory"].append(5)
    copied.fs_mount.append(FsMountOptions(path="/tmp/b", mount_path="/mnt/b"))
    copied.image_options["name"] = "other_image"
    copied.network_configs["public_network_whitelist"].append("10.0.0.2")

    assert hint._pure_depends == [True]
    assert hint.expect_resources == {"cpu": 1, "memory": [2, 3]}
    assert [mount.mount_path for mount in hint.fs_mount] == ["/mnt/a"]
    assert hint.image_options == {"name": "image"}
    assert hint.network_configs == {"public_network_whitelist": ["10.0.0.1"]}


def test_operator_does_not_mutate_caller_scheduling_hint():
    hint = SchedulingHint(expect_resources={"cpu": [1]})

    op1 = MyOperator(scheduling_hint=hint)
    op2 = MyOperator(scheduling_hint=hint)

    op1.scheduling_hint.expect_resources["cpu"].append(2)

    assert hint.expect_resources == {"cpu": [1]}
    assert op2.scheduling_hint.expect_resources == {"cpu": [1]}
