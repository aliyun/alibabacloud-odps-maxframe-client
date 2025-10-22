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

import textwrap

from odps import ODPS
from odps.errors import NoSuchObject

from maxframe.tests.utils import tn
from maxframe.udf import ODPSFunction


def test_odps_function():
    func_body = """from odps.udf import annotate
    @annotate("bigint->bigint")
    class MyMul(object):
        def evaluate(self, arg0):
            return arg0 * 2 if arg0 is not None else None"""
    odps_entry = ODPS.from_environments()
    res_name = tn("test_res")
    func_name = tn("test_odps_func")

    def _cleanup():
        try:
            odps_entry.delete_resource(res_name + ".py")
        except NoSuchObject:
            pass
        try:
            odps_entry.delete_function(func_name)
        except NoSuchObject:
            pass

    _cleanup()

    try:
        test_res = odps_entry.create_resource(
            res_name + ".py", "py", fileobj=textwrap.dedent(func_body)
        )
        test_odps_func_obj = odps_entry.create_function(
            func_name, class_type=f"{res_name}.MyMul", resources=[test_res]
        )
        func = ODPSFunction.wrap(test_odps_func_obj)
        assert isinstance(func, ODPSFunction)
        assert func.__name__ == func_name
        assert func.full_function_name in (
            f"{odps_entry.project}:{func_name}",
            f"{odps_entry.project}:default:{func_name}",
        )
    finally:
        _cleanup()
