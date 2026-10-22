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

import re

import pyarrow as pa

from maxframe import opcodes
from maxframe.liteframe.expressions import LiteFrameMultiColumnExpr, LiteFrameUnaryExpr
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
    "find": wrap_arrow_dtype(pa.int64()),
    "count": wrap_arrow_dtype(pa.int64()),
    "match": wrap_arrow_dtype(pa.bool_()),
    "fullmatch": wrap_arrow_dtype(pa.bool_()),
    "isalpha": wrap_arrow_dtype(pa.bool_()),
    "isalnum": wrap_arrow_dtype(pa.bool_()),
    "isdigit": wrap_arrow_dtype(pa.bool_()),
    "isdecimal": wrap_arrow_dtype(pa.bool_()),
    "isnumeric": wrap_arrow_dtype(pa.bool_()),
    "isspace": wrap_arrow_dtype(pa.bool_()),
    "islower": wrap_arrow_dtype(pa.bool_()),
    "isupper": wrap_arrow_dtype(pa.bool_()),
    "istitle": wrap_arrow_dtype(pa.bool_()),
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


# Handler registry — methods implementable via single-column projection.
# Each entry enables the corresponding .str.<method>() on LiteFrame; the
# DPE codegen layer maps the pandas name to its Polars str.* equivalent.
str_method_to_handlers = {
    "len": StrMethodBaseHandler,
    "upper": StrMethodBaseHandler,
    "lower": StrMethodBaseHandler,
    "title": StrMethodBaseHandler,
    "capitalize": StrMethodBaseHandler,
    "strip": StrMethodBaseHandler,
    "lstrip": StrMethodBaseHandler,
    "rstrip": StrMethodBaseHandler,
    "removeprefix": StrMethodBaseHandler,
    "removesuffix": StrMethodBaseHandler,
    "contains": StrMethodBaseHandler,
    "startswith": StrMethodBaseHandler,
    "endswith": StrMethodBaseHandler,
    "match": StrMethodBaseHandler,
    "fullmatch": StrMethodBaseHandler,
    "find": StrMethodBaseHandler,
    "count": StrMethodBaseHandler,
    "repeat": StrMethodBaseHandler,
    "slice": StrMethodBaseHandler,
    "ljust": StrMethodBaseHandler,
    "rjust": StrMethodBaseHandler,
    "center": StrMethodBaseHandler,
    "pad": StrMethodBaseHandler,
    "zfill": StrMethodBaseHandler,
    "normalize": StrMethodBaseHandler,
    "replace": StrMethodBaseHandler,
    "isalnum": StrMethodBaseHandler,
    "isalpha": StrMethodBaseHandler,
    "isdigit": StrMethodBaseHandler,
    "isdecimal": StrMethodBaseHandler,
    "isnumeric": StrMethodBaseHandler,
    "isspace": StrMethodBaseHandler,
    "islower": StrMethodBaseHandler,
    "isupper": StrMethodBaseHandler,
    "istitle": StrMethodBaseHandler,
}


class LiteFrameStrMultiExpr(LiteFrameMultiColumnExpr):
    """Expression for string operations that produce multiple columns.

    Used by partition/rpartition (always 3 columns), extract (N capture
    groups), and split/rsplit with expand=True and n>=0 (n+1 columns).
    """

    _op_type_ = opcodes.STRING_METHOD  # Reuse existing opcode

    method = StringField("method")
    args = TupleField("args", default=())
    kwargs = DictField("kwargs", default={})


class StrMultiColumnHandler:
    """Handler for multi-column string operations.

    Each subclass implements :meth:`infer_dtypes` to return an ordered
    ``{output_name: dtype}`` mapping.  The number of output columns must be
    determinable at graph-build time (from the method name and/or arguments).
    """

    @staticmethod
    def infer_dtypes(method: str, input_dtype, *args, **kwargs) -> dict:
        raise NotImplementedError


class PartitionHandler(StrMultiColumnHandler):
    """Handler for partition/rpartition — always 3 string columns [0, 1, 2]."""

    @staticmethod
    def infer_dtypes(method, input_dtype, *args, **kwargs):
        sep = args[0] if args else kwargs.get("sep", " ")
        if sep == "":
            raise ValueError("empty separator")
        return {0: _STR_DTYPE, 1: _STR_DTYPE, 2: _STR_DTYPE}


class ExtractHandler(StrMultiColumnHandler):
    """Handler for extract — N columns where N = number of capture groups.

    pandas ``extract(expand=False)`` returns a Series when the pattern has
    exactly one capture group, and a DataFrame (one column per group) when
    it has two or more.  ``expand=True`` always returns a DataFrame.  So
    only the single-group case takes the single-column path; multi-group
    patterns go multi-column regardless of ``expand``.
    """

    @staticmethod
    def infer_dtypes(method, input_dtype, *args, **kwargs):
        expand = kwargs.get("expand", True)
        pat = args[0] if args else kwargs.get("pat")
        if pat is None:
            raise ValueError("extract requires a pattern with capture groups")
        n_groups = re.compile(pat).groups
        if n_groups == 0:
            raise ValueError(
                "pattern contains no capture groups; "
                "add a capture group or use extractall"
            )
        # Single group + expand=False → single-column Series (pandas).
        # Otherwise (multi-group, or expand=True) → one column per group.
        if n_groups == 1 and not expand:
            return {}
        return {i: _STR_DTYPE for i in range(n_groups)}


class SplitHandler(StrMultiColumnHandler):
    """Handler for split/rsplit.

    * ``expand=False`` (default): single list-of-string column → ``{}``
      so the accessor falls back to the single-column path.
    * ``expand=True, n>0``: ``n+1`` string columns, statically known.
    * ``expand=True, n<=0`` (including the default ``-1`` and ``0``):
      pandas treats ``n<=0`` as unlimited, so the column count depends on
      the runtime data.  Reject at graph-build time rather than silently
      degrading to a single list column that fails (or misleads) downstream.
    """

    @staticmethod
    def infer_dtypes(method, input_dtype, *args, **kwargs):
        expand = kwargs.get("expand", False)
        if not expand:
            return {}
        n = kwargs.get("n", -1)
        if n <= 0:
            raise ValueError(
                f"{method}(expand=True, n={n}) is not supported: the output "
                f"column count depends on the data and cannot be determined "
                f"at build time. Use expand=True with n>0, or expand=False "
                f"for a list-of-string column."
            )
        return {i: _STR_DTYPE for i in range(n + 1)}


# Handler registry for multi-column string methods.  Each entry enables the
# corresponding .str.<method>() on LiteFrame via the multi-column projection
# path (LiteFrameMultiColumnExpr + LiteFrameMultiColumnSubExpr).
str_multi_method_to_handlers = {
    "partition": PartitionHandler,
    "rpartition": PartitionHandler,
    "extract": ExtractHandler,
    "split": SplitHandler,
    "rsplit": SplitHandler,
}
