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
import pyarrow as pa

from maxframe.liteframe.accessors._docstring import (
    _adapt_liteframe_docstring,
    _copy_signature,
)
from maxframe.liteframe.accessors.string_.core import (
    LiteFrameStrExpr,
    LiteFrameStrMultiExpr,
    StrMethodBaseHandler,
    str_method_to_handlers,
    str_multi_method_to_handlers,
)
from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _resolve_column_expr,
)
from maxframe.liteframe.expressions import LiteFrameMultiColumnSubExpr
from maxframe.utils import wrap_arrow_dtype


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

    def _build_single_str_projection(self, method, args, kwargs, dtype):
        """Build a fused Projection wrapping a single-column LiteFrameStrExpr."""
        col_name = self._liteframe.columns[0]
        col_dtype = self._liteframe.dtypes.iloc[0]
        col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)
        expr = LiteFrameStrExpr(
            operand=col_expr,
            method=method,
            args=args,
            kwargs=kwargs,
            dtype=dtype,
        )
        named_expr = expr.rename(col_name)
        return _build_fused_projection(self._liteframe, [named_expr])

    @classmethod
    def _gen_func(cls, method):
        """Generate method wrapper with dtype inference."""

        def _inner(self, *args, **kwargs):
            handler = str_method_to_handlers.get(method, StrMethodBaseHandler)
            dtype = handler.infer_dtype(method, self._liteframe.dtypes.iloc[0])
            return self._build_single_str_projection(method, args, kwargs, dtype)

        pd_method = getattr(pd.Series.str, method, None)
        if pd_method is not None:
            _inner.__doc__ = _adapt_liteframe_docstring(pd_method.__doc__)
        return _copy_signature(_inner, pd.Series.str, method)

    @classmethod
    def _gen_multi_func(cls, method):
        """Generate method wrapper for multi-column string operations.

        Inspects the call arguments to determine whether the result is
        single-column (falls back to the base path) or multi-column (uses
        the LiteFrameMultiColumnExpr projection path).
        """

        def _inner(self, *args, **kwargs):
            handler = str_multi_method_to_handlers[method]
            input_dtype = self._liteframe.dtypes.iloc[0]
            dtypes_map = handler.infer_dtypes(method, input_dtype, *args, **kwargs)

            # If the handler returns an empty mapping, the call produces a
            # single-column result.  split/rsplit with expand=False returns a
            # list-of-string column (pandas semantics); split with expand=True
            # and n=-1 is rejected by the handler (dynamic column count).
            # extract with expand=False is a single string column.
            if not dtypes_map:
                if method in ("split", "rsplit") and not kwargs.get("expand", False):
                    # List-of-string output; the codegen emits str.split/rsplit
                    # and the dtype is a list ArrowDtype.
                    dtype = wrap_arrow_dtype(pa.list_(pa.string()))
                else:
                    base_handler = str_method_to_handlers.get(
                        method, StrMethodBaseHandler
                    )
                    dtype = base_handler.infer_dtype(method, input_dtype)
                return self._build_single_str_projection(method, args, kwargs, dtype)

            output_names = [str(k) for k in dtypes_map.keys()]
            col_name = self._liteframe.columns[0]
            col_dtype = self._liteframe.dtypes.iloc[0]
            col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)

            multi_expr = LiteFrameStrMultiExpr(
                operands=[col_expr],
                method=method,
                args=args,
                kwargs=kwargs,
                output_names=output_names,
                dtypes={str(k): str(v) for k, v in dtypes_map.items()},
            )

            projections = []
            for key, name in zip(dtypes_map.keys(), output_names):
                sub_expr = LiteFrameMultiColumnSubExpr(
                    multi_expr=multi_expr,
                    output_name=name,
                    dtype=dtypes_map[key],
                )
                named = sub_expr.rename(name)
                projections.append(named)

            return _build_fused_projection(self._liteframe, projections)

        pd_method = getattr(pd.Series.str, method, None)
        if pd_method is not None:
            _inner.__doc__ = _adapt_liteframe_docstring(pd_method.__doc__)
        return _copy_signature(_inner, pd.Series.str, method)

    @classmethod
    def _register_multi(cls, method):
        """Register a multi-column method to accessor class."""
        func = cls._gen_multi_func(method)
        setattr(cls, method, func)

    @classmethod
    def _register(cls, method):
        """Register method to accessor class."""
        setattr(cls, method, cls._gen_func(method))
