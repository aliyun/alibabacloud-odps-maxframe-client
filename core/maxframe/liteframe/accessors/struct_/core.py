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

from maxframe import opcodes
from maxframe.liteframe.expressions import LiteFrameUnaryExpr
from maxframe.serialization.serializables import DictField, StringField, TupleField
from maxframe.utils import wrap_arrow_dtype


class LiteFrameStructExpr(LiteFrameUnaryExpr):
    """Expression for struct method operations on LiteFrame columns."""

    _op_type_ = opcodes.SERIES_STRUCT_METHOD

    method = StringField("method")
    args = TupleField("args", default=())
    kwargs = DictField("kwargs", default={})


class StructMethodBaseHandler:
    """Default handler for struct methods."""

    @staticmethod
    def infer_dtype(method, input_dtype, *args, **kwargs):
        """Infer output dtype for this struct method."""
        if method == "field":
            name_or_index = args[0] if args else kwargs.get("name_or_index")
            names = (
                name_or_index if isinstance(name_or_index, list) else [name_or_index]
            )
            pa_type = input_dtype.pyarrow_dtype
            for n in names:
                pa_type = pa_type[n].type
            return wrap_arrow_dtype(pa_type)
        return input_dtype


# Handler registry
struct_method_to_handlers = {
    "field": StructMethodBaseHandler,
}
