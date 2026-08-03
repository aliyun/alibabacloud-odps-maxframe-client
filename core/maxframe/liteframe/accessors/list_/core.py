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

import pyarrow as pa

from maxframe import opcodes
from maxframe.liteframe.expressions import LiteFrameUnaryExpr
from maxframe.serialization.serializables import DictField, StringField, TupleField
from maxframe.utils import wrap_arrow_dtype


class LiteFrameListExpr(LiteFrameUnaryExpr):
    """Expression for list method operations on LiteFrame columns."""

    _op_type_ = opcodes.SERIES_LIST_METHOD

    method = StringField("method")
    args = TupleField("args", default=())
    kwargs = DictField("kwargs", default={})


class ListMethodBaseHandler:
    """Default handler for list methods with direct Polars mapping."""

    @staticmethod
    def infer_dtype(method, input_dtype):
        """Infer output dtype for this list method."""
        pa_type = input_dtype.pyarrow_dtype
        if method in ("__getitem__", "get"):
            return wrap_arrow_dtype(pa_type.value_type)
        elif method == "len":
            return wrap_arrow_dtype(pa.int64())
        return input_dtype


# Handler registry
list_method_to_handlers = {
    "__getitem__": ListMethodBaseHandler,
    "get": ListMethodBaseHandler,
    "len": ListMethodBaseHandler,
}
