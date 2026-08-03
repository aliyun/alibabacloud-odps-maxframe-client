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

# Default dtype for string-returning methods
_STR_DTYPE = wrap_arrow_dtype(pa.string())

# Dtype override mapping - methods returning non-string types
STR_METHOD_OUTPUT_DTYPE = {
    "contains": wrap_arrow_dtype(pa.bool_()),
    "len": wrap_arrow_dtype(pa.int64()),
    "startswith": wrap_arrow_dtype(pa.bool_()),
    "endswith": wrap_arrow_dtype(pa.bool_()),
    "isalnum": wrap_arrow_dtype(pa.bool_()),
    "isalpha": wrap_arrow_dtype(pa.bool_()),
    "isdigit": wrap_arrow_dtype(pa.bool_()),
    "isnumeric": wrap_arrow_dtype(pa.bool_()),
}


class LiteFrameStrExpr(LiteFrameUnaryExpr):
    """Expression for string method operations on LiteFrame columns."""

    _op_type_ = opcodes.STRING_METHOD  # Reuse existing opcode

    method = StringField("method")
    args = TupleField("args", default=())
    kwargs = DictField("kwargs", default={})


class StrMethodHandler:
    """Base handler for str method dtype inference."""

    @staticmethod
    def infer_dtype(method: str, input_dtype):
        """Infer output dtype for this str method."""
        return STR_METHOD_OUTPUT_DTYPE.get(method, _STR_DTYPE)


class StrMethodBaseHandler(StrMethodHandler):
    """Default handler for standard str methods with direct Polars mapping."""

    pass


class StrLenHandler(StrMethodHandler):
    """Handler for str.len() - Polars uses .len_chars() not .len()."""

    pass


# Minimal handler registry - only methods with Polars API differences
str_method_to_handlers = {
    "len": StrLenHandler,
    "upper": StrMethodBaseHandler,
    "contains": StrMethodBaseHandler,
    # All other methods use StrMethodBaseHandler via default in codegen
}
