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

from typing import List, Union

import pandas as pd

from maxframe.liteframe.datatypes import to_arrow_dtypes
from maxframe.liteframe.operators.datasource import LiteFrameFromTensor
from maxframe.liteframe.utils import normalize_col_name
from maxframe.tensor.core import Tensor


def liteframe_from_tensor(
    tensor: Tensor,
    columns: Union[str, List, pd.Index] = None,
    gpu: bool = None,
):
    """Create a LiteFrame from a MaxFrame tensor.

    The tensor may be 1-d or 2-d:

    * 1-d tensor -> a single-column LiteFrame. The column is named
      ``columns[0]`` when *columns* is list-like, or *columns* when it is a
      string. When *columns* is not given, ``"0"`` is used.
    * 2-d tensor (m x n) -> an n-column LiteFrame named by *columns*. When
      *columns* is not given, ``["0", "1", ..., str(n - 1)]`` is used.

    Column names are normalized to lowercase to stay consistent with the
    case-insensitive storage convention used across LiteFrame.

    Parameters
    ----------
    tensor : Tensor
        1-d or 2-d MaxFrame tensor.
    columns : str, list or pandas.Index, optional
        Column name(s) for the resulting LiteFrame.
    gpu : bool, optional
        Whether the result runs on GPU. Inferred from the tensor when omitted.

    Returns
    -------
    LiteFrame
        A lazy LiteFrame backed by the given tensor.

    See Also
    --------
    LiteFrame.to_tensor : Convert the LiteFrame to a MaxFrame tensor.
    LiteFrame : Construct a LiteFrame from dict, DataFrame, or tensor.
    """
    if tensor.ndim not in (1, 2):
        raise TypeError(
            f"Not support creating LiteFrame from {tensor.ndim} dims tensor"
        )

    if tensor.ndim == 1:
        # 1-d tensor -> exactly one column
        if columns is None:
            col_names = ["0"]
        elif isinstance(columns, str):
            col_names = [columns]
        else:
            col_names = [columns[0]]
    else:
        # 2-d tensor (m x n) -> n columns
        n_cols = tensor.shape[1]
        if columns is None:
            col_names = [str(i) for i in range(n_cols)]
        elif isinstance(columns, str):
            raise ValueError("columns must be a list-like object for 2-d tensor")
        else:
            col_names = list(columns)
            if len(col_names) != n_cols:
                raise ValueError(
                    f"columns should have length {n_cols}, got {len(col_names)}"
                )

    # Normalize to lowercase so storage stays case-insensitive, matching
    # the convention enforced by from_local_df.
    col_names = [normalize_col_name(str(c)) for c in col_names]

    # Guard against duplicate column names after normalization.
    seen = set()
    for c in col_names:
        if c in seen:
            raise ValueError(f"Duplicate column name after case normalization: '{c}'")
        seen.add(c)

    dtypes = pd.Series([tensor.dtype] * len(col_names), index=col_names)
    # Match from_local_df which stores Arrow dtypes so the rest of the
    # LiteFrame machinery (dtypes property, arithmetic, ufunc, etc.) stays
    # consistent.
    dtypes = to_arrow_dtypes(dtypes)

    gpu = tensor.op.gpu if gpu is None else gpu
    op = LiteFrameFromTensor(input=tensor, dtypes=dtypes, gpu=gpu)
    return op(tensor)
