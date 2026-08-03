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

from maxframe.liteframe.accessors.string_.core import (
    LiteFrameStrExpr,
    StrMethodBaseHandler,
    str_method_to_handlers,
)
from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _resolve_column_expr,
)


class StringAccessor:
    """
    Vectorized string functions for LiteFrame.

    Provides pandas-style .str accessor for single-column LiteFrame.
    Multi-column LiteFrames raise ValueError on accessor access.

    Examples
    --------
    >>> df = LiteFrame({"name": ["alice", "bob"]})
    >>> df.str.upper().execute()
       name
    0 ALICE
    1   BOB
    """

    def __init__(self, liteframe):
        if len(liteframe.columns) != 1:
            raise ValueError("Cannot apply .str accessor to multi-column LiteFrame")
        self._liteframe = liteframe

    @classmethod
    def _gen_func(cls, method):
        """Generate method wrapper with dtype inference."""

        def _inner(self, *args, **kwargs):
            # Infer dtype via handler
            handler = str_method_to_handlers.get(method, StrMethodBaseHandler)
            dtype = handler.infer_dtype(method, self._liteframe.dtypes.iloc[0])

            # Create column expression for the single column
            col_name = self._liteframe.columns[0]
            col_dtype = self._liteframe.dtypes.iloc[0]
            col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)

            # Create expression
            expr = LiteFrameStrExpr(
                operand=col_expr,
                method=method,
                args=args,
                kwargs=kwargs,
                dtype=dtype,
            )

            # Wrap in NamedExpr with original column name
            named_expr = expr.rename(self._liteframe.columns[0])

            # Create projection operator with the NamedExpr
            return _build_fused_projection(self._liteframe, [named_expr])

        # TODO: Add docstrings later for each method
        return _inner

    @classmethod
    def _register(cls, method):
        """Register method to accessor class."""
        setattr(cls, method, cls._gen_func(method))
