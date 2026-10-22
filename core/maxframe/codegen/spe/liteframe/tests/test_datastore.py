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

import glob as _glob
import os as _os
import tempfile as _tempfile

import pandas as pd
import pyarrow as pa
import pytest

from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.liteframe.datasource import (
    CSVDataSourceHandler,
    LiteFrameReadLanceAdapter,
)
from maxframe.codegen.spe.liteframe.datastore import (
    CSVDataStoreHandler,
    JSONDataStoreHandler,
    LiteFrameLakeDataStoreAdapter,
    LiteFrameToLanceAdapter,
    LiteFrameToODPSTableAdapter,
    ParquetDataStoreHandler,
)
from maxframe.liteframe.operators.core import LiteFrameDataFormat
from maxframe.liteframe.operators.datastore import (
    LiteFrameLakeDataStore,
    LiteFrameToLance,
    LiteFrameToODPSTable,
)
from maxframe.utils.datatypes import is_arrow_dtype_supported

_requires_arrow_dtype = pytest.mark.skipif(
    not is_arrow_dtype_supported(),
    reason="pandas ArrowDtype is not supported in current environment",
)

try:
    import lance as _lance
except (ImportError, AttributeError):  # pragma: no cover
    _lance = None
_requires_lance = pytest.mark.skipif(_lance is None, reason="pylance is not installed")


# ---- code generation ----


def test_to_odps_adapter_code_generation(make_liteframe):
    """LiteFrameToODPSTableAdapter should generate _write_as_table call."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    data = make_liteframe(df)
    op = LiteFrameToODPSTable(
        table_name="test_table",
        mode="overwrite",
        unknown_as_string=True,
        dtypes=pd.Series({"a": df["a"].dtype}),
    )
    out = op(data)
    op = out.op
    adapter = LiteFrameToODPSTableAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 2
    assert "LiteFrameToODPSTableAdapter._write_as_table" in result[0]
    assert "= None" in result[1]


@pytest.mark.parametrize(
    "data_format, handler_class, path, extra_kwargs, expected_substring",
    [
        pytest.param(
            LiteFrameDataFormat.CSV,
            CSVDataStoreHandler,
            "out-*.csv",
            {},
            None,
            id="csv",
        ),
        pytest.param(
            LiteFrameDataFormat.PARQUET,
            ParquetDataStoreHandler,
            "out-*.parquet",
            {"compression": "snappy"},
            "compression='snappy'",
            id="parquet",
        ),
        pytest.param(
            LiteFrameDataFormat.JSON,
            JSONDataStoreHandler,
            "out-*.json",
            {},
            None,
            id="json",
        ),
    ],
)
def test_to_files_adapter_code_generation(
    make_liteframe, data_format, handler_class, path, extra_kwargs, expected_substring
):
    lf = make_liteframe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    op = LiteFrameLakeDataStore(
        path=path,
        data_format=data_format,
        compression=extra_kwargs.pop("compression", None),
        dtypes=lf.dtypes,
        write_kwargs={
            "engine": "pyarrow",
            **extra_kwargs,
        }
        or None,
    )
    out = op(lf)
    adapter = LiteFrameLakeDataStoreAdapter()
    result = adapter.generate_code(out.op, SPECodeContext())
    assert len(result) == 2
    assert f"{handler_class.__name__}.write_as_files" in result[0]
    if expected_substring is not None:
        assert expected_substring in result[0]
    assert result[1].endswith("= None")


@_requires_arrow_dtype
def test_to_csv_partitioned():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "p": ["x", "x", "y", "y"]})
    with _tempfile.TemporaryDirectory() as d:
        base = _os.path.join(d, "out")
        CSVDataStoreHandler.write_as_files(df, base, partition_cols=["p"])
        # Each partition value gets its own subdirectory.
        assert _os.path.isdir(_os.path.join(base, "p=x"))
        assert _os.path.isdir(_os.path.join(base, "p=y"))
        # Read both back and verify the union of rows.
        parts = _glob.glob(_os.path.join(base, "**", "data_0.csv"), recursive=True)
        got = pd.concat([CSVDataSourceHandler.read_as_pandas(p) for p in parts])
    assert sorted(got["a"]) == [1, 2, 3, 4]


def test_to_parquet_unsupported_kwarg_raises():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    with _tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "out-*.parquet")
        with pytest.raises(NotImplementedError, match="does not support kwarg"):
            ParquetDataStoreHandler.write_as_files(
                df, path, write_kwargs={"additional_kwargs": {"bogus": 1}}
            )


@_requires_arrow_dtype
@pytest.mark.parametrize("partition_cols", [None, ["p"]])
def test_to_parquet_statistics_false(partition_cols):
    """polars `statistics` must translate to pyarrow `write_statistics`."""
    df = pd.DataFrame({"a": [1, 2, 3], "p": ["x", "y", "y"]})
    with _tempfile.TemporaryDirectory() as d:
        if partition_cols:
            out = _os.path.join(d, "out")
            ParquetDataStoreHandler.write_as_files(
                df,
                out,
                partition_cols=partition_cols,
                write_kwargs={"statistics": False},
                compression="snappy",
            )
            parts = _glob.glob(_os.path.join(out, "**", "*.parquet"), recursive=True)
        else:
            out = _os.path.join(d, "out.parquet")
            ParquetDataStoreHandler.write_data(
                df,
                out,
                compression="snappy",
                write_kwargs={"statistics": False},
            )
            parts = [out]
    assert parts


# ---- Lance ----


def test_to_lance_adapter_code_generation(make_liteframe):
    lf = make_liteframe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    op = LiteFrameToLance(path="out.lance", dtypes=lf.dtypes, mode="overwrite")
    out = op(lf)
    adapter = LiteFrameToLanceAdapter()
    result = adapter.generate_code(out.op, SPECodeContext())
    assert len(result) == 2
    assert "LiteFrameToLanceAdapter._write_as_lance" in result[0]
    assert "mode='overwrite'" in result[0]
    assert result[1].endswith("= None")


@_requires_arrow_dtype
@_requires_lance
@pytest.mark.parametrize(
    "modes, expected_len, expected_a, check_columns",
    [
        pytest.param(["overwrite"], 3, [1, 2, 3], True, id="overwrite"),
        pytest.param(
            ["overwrite", "append"], 6, [1, 2, 3, 1, 2, 3], False, id="append"
        ),
    ],
)
def test_to_lance_round_trip(
    io_test_df, modes, expected_len, expected_a, check_columns
):
    with _tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "out.lance")
        for mode in modes:
            LiteFrameToLanceAdapter._write_as_lance(io_test_df, path, mode=mode)
        got = LiteFrameReadLanceAdapter._read_as_pandas(path)
    assert len(got) == expected_len
    if check_columns:
        assert list(got.columns) == ["a", "b", "c"]
    assert list(got["a"]) == expected_a


@_requires_arrow_dtype
@_requires_lance
def test_to_lance_schema_override(io_test_df):
    """schema= re-types the named input columns before writing."""
    schema = pa.schema([pa.field("a", pa.int32())])
    with _tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "out.lance")
        LiteFrameToLanceAdapter._write_as_lance(
            io_test_df,
            path,
            mode="overwrite",
            schema_bytes=schema.serialize().to_pybytes(),
        )
        got = LiteFrameReadLanceAdapter._read_as_pandas(path)
    assert str(got["a"].dtype) == "int32[pyarrow]"


@_requires_arrow_dtype
@_requires_lance
def test_to_lance_merge_mode_not_supported(io_test_df):
    """Merge mode requires distributed fragment alignment (DPE only)."""
    with _tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "out.lance")
        with pytest.raises(NotImplementedError, match="merge"):
            LiteFrameToLanceAdapter._write_as_lance(io_test_df, path, mode="merge")
