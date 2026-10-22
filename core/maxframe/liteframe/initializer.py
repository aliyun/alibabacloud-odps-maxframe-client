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

from maxframe.liteframe.core import LiteFrame as _LiteFrame
from maxframe.serialization.serializables import SerializableMeta
from maxframe.tensor.core import TENSOR_TYPE


class InitializerMeta(SerializableMeta):
    def __instancecheck__(cls, instance):
        return isinstance(
            instance, (cls.__base__,) + getattr(cls, "_allow_data_type_", ())
        )


class LiteFrame(_LiteFrame, metaclass=InitializerMeta):
    """A performance-dedicated, lazy, distributed tabular data object with no index semantics.

    LiteFrame provides a pandas-looking DataFrame API for large-scale data
    processing on MaxCompute. Unlike a traditional DataFrame, LiteFrame has
    no row index: all operations are column-based, single-column selection
    returns a LiteFrame (never a Series), and there is no implicit label
    alignment between independent frames. Arrow dtypes are used as
    first-class citizens so that integer columns with nulls stay nullable
    integer rather than being silently promoted to float.

    By removing index overhead and embedding element-wise expressions in a
    small set of fusible operators, LiteFrame achieves lower planning cost
    and higher execution throughput.

    Operations are lazy — they build a computation graph that is executed
    on demand via ``.execute()``. Row order is preserved through execution
    unless an operator explicitly reorders rows.

    Parameters
    ----------
    data : dict, pandas.DataFrame, tensor, or LiteFrame
        Input data. A ``dict`` or ``pandas.DataFrame`` is converted
        internally to an Arrow-backed LiteFrame. A tensor is wrapped as
        a single-column (or multi-column) frame. An existing LiteFrame is
        returned as-is.
    columns : list of str, optional
        Column labels to use. When ``data`` is a dict or DataFrame,
        this selects and orders the output columns.
    dtype : dtype, optional
        Data type to force on all columns.
    chunk_size : int, optional
        Row-axis chunk size for distributed execution.
    gpu : bool, optional
        Whether to use GPU execution.

    See Also
    --------
    maxframe.liteframe.read_csv : Read a CSV file into a LiteFrame.
    maxframe.liteframe.read_parquet : Read a Parquet file into a LiteFrame.
    maxframe.liteframe.concat : Concatenate LiteFrames along the row axis.
    maxframe.liteframe.LiteFrame.from_tensor : Create a LiteFrame from a MaxFrame tensor.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> lf.execute()
       a  b
    0  1  4
    1  2  5
    2  3  6
    """

    def __init__(
        self,
        data=None,
        columns=None,
        dtype=None,
        chunk_size=None,
        gpu=None,
    ):
        # Lazy import to avoid circular dependency with datasource module
        from maxframe.liteframe.datasource.from_local import from_local_df
        from maxframe.liteframe.datasource.from_tensor import liteframe_from_tensor

        if isinstance(data, _LiteFrame):
            super().__init__(data.data)
            return

        if isinstance(data, TENSOR_TYPE):
            tensor = data
            # LiteFrame is only chunked along the row axis; when a chunk size is
            # given, rechunk the rows of the tensor accordingly.
            if chunk_size is not None:
                tensor = tensor.rechunk({0: chunk_size})
            lf = liteframe_from_tensor(tensor, columns=columns, gpu=gpu)
            if dtype is not None:
                lf = lf.astype(dtype)
            super().__init__(lf.data)
            return

        if isinstance(data, dict):
            pdf = pd.DataFrame(data, columns=columns, dtype=dtype)
        elif isinstance(data, pd.DataFrame):
            pdf = data.copy()
            if columns is not None:
                pdf = pdf[columns]
        elif data is None:
            pdf = pd.DataFrame(columns=columns or [], dtype=dtype)
        else:
            pdf = pd.DataFrame(data, columns=columns, dtype=dtype)

        lf = from_local_df(pdf, chunk_size=chunk_size, gpu=gpu)
        super().__init__(lf.data)
