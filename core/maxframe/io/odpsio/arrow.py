# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

from typing import Any, Tuple, Union

import pandas as pd
import pyarrow as pa

from ...core import OutputType
from ...protocol import DataFrameTableMeta
from ...tensor.core import TENSOR_TYPE
from ...typing_ import ArrowTableType, PandasObjectTypes
from .schema import build_dataframe_table_meta


def _rebuild_dataframe(
    df: pd.DataFrame, table_meta: DataFrameTableMeta
) -> Union[pd.DataFrame, pd.Series]:
    indexed = df.set_index(list(table_meta.table_index_column_names))
    indexed.index.set_names(table_meta.pd_index_level_names, inplace=True)
    if len(table_meta.pd_column_level_names) == 1:
        cols = pd.Index(
            table_meta.pd_column_names, name=table_meta.pd_column_level_names[0]
        )
    else:
        cols = pd.MultiIndex.from_tuples(table_meta.pd_column_names).set_names(
            table_meta.pd_column_level_names
        )
    indexed.columns = cols
    if table_meta.type == OutputType.series:
        return indexed.iloc[:, 0]
    return indexed


def _rebuild_index(df: pd.DataFrame, table_meta: DataFrameTableMeta) -> pd.Index:
    if df.shape[1] > 1:
        idx = pd.MultiIndex.from_frame(df)
        idx.names = table_meta.pd_index_level_names
    else:
        # make sure even if None names are updated properly
        idx = pd.Index(df.iloc[:, 0])
        idx.name = table_meta.pd_index_level_names[0]
    return idx


def arrow_to_pandas(
    arrow_table: ArrowTableType, table_meta: DataFrameTableMeta
) -> PandasObjectTypes:
    df = arrow_table.to_pandas()
    if table_meta.type in (OutputType.dataframe, OutputType.series):
        return _rebuild_dataframe(df, table_meta)
    elif table_meta.type == OutputType.index:
        return _rebuild_index(df, table_meta)
    elif table_meta.type == OutputType.scalar:
        return _rebuild_index(df, table_meta)[0]
    else:  # this could never happen  # pragma: no cover
        raise ValueError(f"Does not support meta type {table_meta.type!r}")


def pandas_to_arrow(
    df: Any, nthreads=1, ignore_index=False
) -> Tuple[ArrowTableType, DataFrameTableMeta]:
    table_meta = build_dataframe_table_meta(df, ignore_index)
    df = df.copy() if callable(getattr(df, "copy", None)) else df
    if table_meta.type in (OutputType.dataframe, OutputType.series):
        if table_meta.type == OutputType.series:
            df = df.to_frame("_data" if df.name is None else df.name)
        df.columns = pd.Index(table_meta.table_column_names)
        if not ignore_index:
            df = df.rename_axis(table_meta.table_index_column_names).reset_index()
    elif ignore_index and table_meta.type != OutputType.index:
        df = pd.DataFrame([], columns=[])
    elif table_meta.type == OutputType.index:
        names = [f"_idx_{idx}" for idx in range(len(df.names))]
        df = df.to_frame(name=names[0] if len(names) == 1 else names)
    elif table_meta.type == OutputType.scalar:
        names = ["_idx_0"]
        if isinstance(df, TENSOR_TYPE):
            df = pd.DataFrame([], columns=names).astype({names[0]: df.dtype})
        else:
            df = pd.DataFrame([[df]], columns=names)
    else:  # this could never happen  # pragma: no cover
        raise ValueError(f"Does not support meta type {table_meta.type!r}")
    return pa.Table.from_pandas(df, nthreads=nthreads, preserve_index=False), table_meta
