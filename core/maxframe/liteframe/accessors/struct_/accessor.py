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

import pandas as pd

from maxframe.lib.dtypes_extension import is_struct_dtype
from maxframe.liteframe.accessors.struct_.core import (
    LiteFrameStructExpr,
    StructMethodBaseHandler,
    struct_method_to_handlers,
)
from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _resolve_column_expr,
)
from maxframe.utils import wrap_arrow_dtype


class StructAccessor:
    """
    Vectorized struct functions for LiteFrame.

    Provides pandas-style .struct accessor for single-column LiteFrame with
    Arrow struct dtype. Multi-column LiteFrames raise ValueError on accessor
    access.

    Examples
    --------
    >>> df = LiteFrame({"s": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]})
    >>> df.struct.field("a").execute()
       a
    0  1
    1  2
    """

    def __init__(self, liteframe):
        if len(liteframe.columns) != 1:
            raise ValueError("Cannot apply .struct accessor to multi-column LiteFrame")
        dtype = liteframe.dtypes.iloc[0]
        if not is_struct_dtype(dtype):
            raise AttributeError("Can only use .struct accessor with struct values")
        self._liteframe = liteframe

    @classmethod
    def _gen_func(cls, method):
        """Generate method wrapper with dtype inference."""

        def _inner(self, *args, **kwargs):
            handler = struct_method_to_handlers.get(method, StructMethodBaseHandler)
            dtype = handler.infer_dtype(
                method, self._liteframe.dtypes.iloc[0], *args, **kwargs
            )

            col_name = self._liteframe.columns[0]
            col_dtype = self._liteframe.dtypes.iloc[0]
            col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)

            expr = LiteFrameStructExpr(
                operand=col_expr,
                method=method,
                args=args,
                kwargs=kwargs,
                dtype=dtype,
            )

            # Resolve field name for output column name
            if method == "field":
                name_or_index = args[0] if args else kwargs.get("name_or_index")
                names = (
                    name_or_index
                    if isinstance(name_or_index, list)
                    else [name_or_index]
                )
                pa_type = self._liteframe.dtypes.iloc[0].pyarrow_dtype
                out_name = None
                for n in names:
                    out_name = pa_type[n].name
                    pa_type = pa_type[n].type
                named_expr = expr.rename(out_name)
            else:
                named_expr = expr.rename(self._liteframe.columns[0])

            return _build_fused_projection(self._liteframe, [named_expr])

        return _inner

    @property
    def dtypes(self):
        """Return the dtype object of each child field of the struct.

        Returns a plain ``pandas.Series`` of dtype objects indexed by
        field names — direct metadata extraction, no LiteFrame operation.
        """
        pa_type = self._liteframe.dtypes.iloc[0].pyarrow_dtype
        fields = [pa_type[i] for i in range(pa_type.num_fields)]
        return pd.Series(
            [wrap_arrow_dtype(f.type) for f in fields],
            index=[f.name for f in fields],
        )

    @classmethod
    def _register(cls, method):
        """Register method to accessor class."""
        setattr(cls, method, cls._gen_func(method))
