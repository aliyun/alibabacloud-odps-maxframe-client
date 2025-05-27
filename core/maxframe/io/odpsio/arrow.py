# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

import numpy as np
import pandas as pd
import pyarrow as pa

from ...core import OutputType
from ...lib.version import parse as parse_version
from ...protocol import DataFrameTableMeta
from ...tensor.core import TENSOR_TYPE
from ...typing_ import ArrowTableType, PandasObjectTypes
from .schema import (
    arrow_table_to_pandas_dataframe,
    build_dataframe_table_meta,
    pandas_dataframe_to_arrow_table,
)


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
    df = arrow_table_to_pandas_dataframe(arrow_table, table_meta)
    if table_meta.type in (OutputType.dataframe, OutputType.series):
        return _rebuild_dataframe(df, table_meta)
    elif table_meta.type == OutputType.index:
        return _rebuild_index(df, table_meta)
    elif table_meta.type == OutputType.tensor:
        return _rebuild_index(df, table_meta).to_numpy()
    elif table_meta.type == OutputType.scalar:
        return _rebuild_index(df, table_meta)[0]
    else:  # this could never happen  # pragma: no cover
        raise ValueError(f"Does not support meta type {table_meta.type!r}")


def pandas_to_arrow(
    df: Any, nthreads=1, ignore_index=False, ms_cols=None
) -> Tuple[ArrowTableType, DataFrameTableMeta]:
    table_meta = build_dataframe_table_meta(df, ignore_index)
    df = df.copy() if callable(getattr(df, "copy", None)) else df
    table_datetime_cols = None
    if table_meta.type in (OutputType.dataframe, OutputType.series):
        if table_meta.type == OutputType.series:
            df = df.to_frame("_data" if df.name is None else df.name)
            if ms_cols:
                table_datetime_cols = {"_data"}
        elif ms_cols:
            ms_col_set = set(ms_cols)
            table_datetime_cols = set()
            for pd_col, table_col in zip(
                table_meta.pd_column_dtypes.keys(), table_meta.table_column_names
            ):
                if pd_col in ms_col_set:
                    table_datetime_cols.add(table_col)
        df.columns = pd.Index(table_meta.table_column_names)
        if not ignore_index:
            df = df.rename_axis(table_meta.table_index_column_names).reset_index()
    elif ignore_index and table_meta.type != OutputType.index:
        df = pd.DataFrame([], columns=[])
    elif table_meta.type == OutputType.index:
        names = [f"_idx_{idx}" for idx in range(len(df.names))]
        table_datetime_cols = set()
        if ms_cols:
            if isinstance(df, pd.MultiIndex):
                table_datetime_cols = {f"_idx_{idx}" for idx in ms_cols}
            else:
                table_datetime_cols = {"_idx_0"}
        df = df.to_frame(name=names[0] if len(names) == 1 else names)
    elif table_meta.type in (OutputType.scalar, OutputType.tensor):
        names = ["_idx_0"]
        if isinstance(df, (TENSOR_TYPE, np.ndarray)):
            df = pd.DataFrame([], columns=names).astype({names[0]: df.dtype})
        else:
            df = pd.DataFrame([[df]], columns=names)
    else:  # this could never happen  # pragma: no cover
        raise ValueError(f"Does not support meta type {table_meta.type!r}")

    try:
        pa_table = pandas_dataframe_to_arrow_table(df, nthreads=nthreads)
    except pa.ArrowTypeError as ex:  # pragma: no cover
        late_np_version = parse_version(np.__version__) >= parse_version("1.20")
        early_pa_version = parse_version(pa.__version__) <= parse_version("4.0")
        if (
            late_np_version
            and early_pa_version
            and "Did not pass numpy.dtype object" in str(ex)
        ):
            raise TypeError(
                "Potential dependency conflict. Try update to pyarrow>4.0 "
                "or downgrade to numpy<1.20. Details can be seen at "
                "https://github.com/numpy/numpy/issues/17913. "
                f"Raw error message: {ex!r}"
            ).with_traceback(ex.__traceback__) from None
        else:
            raise

    if table_datetime_cols:
        col_names = pa_table.schema.names
        col_datas = []
        for idx, col_name in enumerate(pa_table.schema.names):
            if col_name not in table_datetime_cols:
                col_datas.append(pa_table.column(idx))
                continue
            col_data = pa_table.column(idx).cast(pa.timestamp("ms"))
            col_datas.append(col_data)
        pa_table = pa.Table.from_arrays(col_datas, names=col_names)

        new_names, new_dtypes = [], []
        for table_col, (pd_col, pd_dtype) in zip(
            table_meta.table_column_names, table_meta.pd_column_dtypes.items()
        ):
            new_names.append(pd_col)
            if table_col not in table_datetime_cols:
                new_dtypes.append(pd_dtype)
            else:
                new_dtypes.append(np.dtype("datetime64[ms]"))
        table_meta.pd_column_dtypes = pd.Series(new_dtypes, index=new_names)
    return pa_table, table_meta
