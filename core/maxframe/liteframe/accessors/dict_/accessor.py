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

from maxframe.lib.dtypes_extension import is_map_dtype
from maxframe.liteframe.accessors.dict_.core import (
    DictMethodBaseHandler,
    LiteFrameDictExpr,
    dict_method_to_handlers,
)
from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _resolve_column_expr,
)


class DictAccessor:
    """
    Vectorized dict functions for LiteFrame.

    Provides pandas-style .dict accessor for single-column LiteFrame with
    Arrow map dtype. Multi-column LiteFrames raise ValueError on accessor
    access.

    Examples
    --------
    >>> df = LiteFrame({"d": [("k1", 1), ("k2", 2)]})
    >>> df.dict.len().execute()
       d
    0  2
    """

    def __init__(self, liteframe):
        if len(liteframe.columns) != 1:
            raise ValueError("Cannot apply .dict accessor to multi-column LiteFrame")
        dtype = liteframe.dtypes.iloc[0]
        if not is_map_dtype(dtype):
            raise AttributeError("Can only use .dict accessor with dict values")
        self._liteframe = liteframe

    @classmethod
    def _gen_func(cls, method):
        """Generate method wrapper with dtype inference."""

        def _inner(self, *args, **kwargs):
            handler = dict_method_to_handlers.get(method, DictMethodBaseHandler)
            dtype = handler.infer_dtype(method, self._liteframe.dtypes.iloc[0])

            col_name = self._liteframe.columns[0]
            col_dtype = self._liteframe.dtypes.iloc[0]
            col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)

            expr = LiteFrameDictExpr(
                operand=col_expr,
                method=method,
                args=args,
                kwargs=kwargs,
                dtype=dtype,
            )

            named_expr = expr.rename(self._liteframe.columns[0])
            return _build_fused_projection(self._liteframe, [named_expr])

        return _inner

    @classmethod
    def _register(cls, method):
        """Register method to accessor class."""
        setattr(cls, method, cls._gen_func(method))
