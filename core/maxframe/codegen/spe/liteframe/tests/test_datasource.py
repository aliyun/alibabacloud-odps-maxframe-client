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

import json as _json
import os as _os
import tempfile as _tempfile
from contextlib import contextmanager

import mock
import pandas as pd
import pyarrow as pa
import pytest

from maxframe.codegen.spe.core import SPECodeContext, SPECodeGenerator
from maxframe.codegen.spe.liteframe import datasource as _ds
from maxframe.codegen.spe.liteframe.datasource import (
    CSVDataSourceHandler,
    JSONDataSourceHandler,
    LiteFrameFromLocalAdapter,
    LiteFrameLakeDataSourceAdapter,
    LiteFrameReadLanceAdapter,
    LiteFrameReadODPSTableAdapter,
    ParquetDataSourceHandler,
)
from maxframe.codegen.spe.liteframe.datastore import (
    CSVDataStoreHandler,
    JSONDataStoreHandler,
    LiteFrameToLanceAdapter,
    ParquetDataStoreHandler,
)
from maxframe.liteframe.datasource.read_csv import read_csv
from maxframe.liteframe.datasource.read_lance import read_lance
from maxframe.liteframe.operators.core import LiteFrameDataFormat
from maxframe.liteframe.operators.datasource import (
    LiteFrameFromLocal,
    LiteFrameLakeDataSource,
    LiteFrameReadLance,
    LiteFrameReadODPSTable,
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


# Shared write lambdas for round-trip and read-dtypes parametrized tests.
_SPE_WRITE_FNS = {
    "csv": lambda df, d: CSVDataStoreHandler.write_as_files(
        df, _os.path.join(d, "out-*.csv")
    ),
    "parquet": lambda df, d: ParquetDataStoreHandler.write_as_files(
        df, _os.path.join(d, "out-*.parquet"), compression="snappy"
    ),
    "json": lambda df, d: JSONDataStoreHandler.write_as_files(
        df, _os.path.join(d, "out-*.json"), lines=True
    ),
    "lance": lambda df, d: LiteFrameToLanceAdapter._write_as_lance(
        df, _os.path.join(d, "out.lance")
    ),
}


@contextmanager
def _temp_io_path(ext):
    """Yield ``(glob_write_path, resolved_read_path)`` for file IO in a temp dir."""
    with _tempfile.TemporaryDirectory() as d:
        yield _os.path.join(d, f"out-*.{ext}"), _os.path.join(d, f"out-0.{ext}")


# ---- code generation ----


@_requires_arrow_dtype
def test_from_local_adapter():
    """LiteFrameFromLocalAdapter should register data as constant."""
    data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    op = LiteFrameFromLocal(
        data=data,
        dtypes=pd.Series(
            [pd.ArrowDtype(pa.int64()), pd.ArrowDtype(pa.int64())], index=["a", "b"]
        ),
        nrows=3,
    )
    tileable = op._new_liteframe_from_source(shape=(3, 2))
    # Use the op from the created tileable to ensure outputs are populated
    op = tileable.op
    adapter = LiteFrameFromLocalAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 1
    assert "const_" in result[0]
    assert "= const_" in result[0]


@_requires_arrow_dtype
def test_read_odps_adapter_code_generation():
    """LiteFrameReadODPSTableAdapter should generate code calling _read_as_pandas."""
    op = LiteFrameReadODPSTable(
        table_name="test_table",
        dtypes=pd.Series([pd.ArrowDtype(pa.int64())], index=["a"]),
    )
    tileable = op._new_liteframe_from_source()
    # Use the op from the created tileable to ensure outputs are populated
    op = tileable.op
    adapter = LiteFrameReadODPSTableAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 1
    assert "LiteFrameReadODPSTableAdapter._read_as_pandas" in result[0]
    assert "test_table" in result[0]


@pytest.mark.parametrize(
    "op_class, adapter_class, path, op_kwargs, expected_substrings",
    [
        pytest.param(
            LiteFrameLakeDataSource,
            LiteFrameLakeDataSourceAdapter,
            "/tmp/x.csv",
            {"data_format": LiteFrameDataFormat.CSV},
            ["CSVDataSourceHandler.read_as_pandas", "'/tmp/x.csv'"],
            id="csv",
        ),
        pytest.param(
            LiteFrameLakeDataSource,
            LiteFrameLakeDataSourceAdapter,
            "/tmp/x.parquet",
            {
                "data_format": LiteFrameDataFormat.PARQUET,
                "columns": ["a"],
                "filters": [("a", ">", 1)],
            },
            ["ParquetDataSourceHandler.read_as_pandas", "columns=['a']"],
            id="parquet",
        ),
        pytest.param(
            LiteFrameLakeDataSource,
            LiteFrameLakeDataSourceAdapter,
            "/tmp/x.json",
            {"data_format": LiteFrameDataFormat.JSON},
            ["JSONDataSourceHandler.read_as_pandas"],
            id="json",
        ),
        pytest.param(
            LiteFrameReadLance,
            LiteFrameReadLanceAdapter,
            "/tmp/x.lance",
            {"columns": ["a"]},
            ["LiteFrameReadLanceAdapter._read_as_pandas", "columns=['a']"],
            id="lance",
        ),
    ],
)
def test_read_adapter_code_generation(
    op_class, adapter_class, path, op_kwargs, expected_substrings
):
    op = op_class(path=path, dtypes=pd.Series(dtype=object), **op_kwargs)
    lf = op._new_liteframe_from_source()
    adapter = adapter_class()
    result = adapter.generate_code(lf.op, SPECodeContext())
    assert len(result) == 1
    for substr in expected_substrings:
        assert substr in result[0]


@pytest.mark.parametrize(
    "op_class, adapter_class, path, op_kwargs, handler_name",
    [
        pytest.param(
            LiteFrameLakeDataSource,
            LiteFrameLakeDataSourceAdapter,
            "/tmp/x.csv",
            {"data_format": LiteFrameDataFormat.CSV},
            "CSVDataSourceHandler",
            id="csv",
        ),
        pytest.param(
            LiteFrameReadLance,
            LiteFrameReadLanceAdapter,
            "/tmp/x.lance",
            {},
            "LiteFrameReadLanceAdapter",
            id="lance",
        ),
    ],
)
def test_read_adapter_get_dtypes_code_generation(
    op_class, adapter_class, path, op_kwargs, handler_name
):
    op = op_class(
        path=path, dtypes=pd.Series(dtype=object), read_stage="get_dtypes", **op_kwargs
    )
    scalar = op._new_liteframe_from_source()
    adapter = adapter_class()
    result = adapter.generate_code(scalar.op, SPECodeContext())
    assert len(result) == 1
    assert "json.dumps" in result[0]
    assert (
        f"{handler_name}.read_dtypes" in result[0]
        or f"{handler_name}._read_dtypes" in result[0]
    )


# ---- static helper round-trips ----


@pytest.mark.parametrize(
    "fmt, read_fn, check_c",
    [
        pytest.param(
            "csv",
            lambda d: CSVDataSourceHandler.read_as_pandas(
                _os.path.join(d, "out-0.csv")
            ),
            True,
            id="csv",
        ),
        pytest.param(
            "parquet",
            lambda d: ParquetDataSourceHandler.read_as_pandas(
                _os.path.join(d, "out-0.parquet")
            ),
            False,
            id="parquet",
        ),
        pytest.param(
            "json",
            lambda d: JSONDataSourceHandler.read_as_pandas(
                _os.path.join(d, "out-0.json"), read_kwargs={"lines": True}
            ),
            False,
            id="json",
        ),
        pytest.param(
            "lance",
            lambda d: LiteFrameReadLanceAdapter._read_as_pandas(
                _os.path.join(d, "out.lance")
            ),
            False,
            marks=[_requires_lance],
            id="lance",
        ),
    ],
)
@_requires_arrow_dtype
def test_round_trip(io_test_df, fmt, read_fn, check_c):
    write_fn = _SPE_WRITE_FNS[fmt]
    with _tempfile.TemporaryDirectory() as d:
        write_fn(io_test_df, d)
        got = read_fn(d)
    assert list(got.columns) == ["a", "b", "c"]
    assert list(got["a"]) == [1, 2, 3]
    assert str(got["a"].dtype) == "int64[pyarrow]"
    if check_c:
        assert list(got["c"]) == ["x", "y", "z"]


@_requires_arrow_dtype
def test_read_json_default_head_bytes_not_forwarded(io_test_df):
    """head_bytes must not reach pd.read_json on the default read_json path (#237871464)."""
    with _tempfile.TemporaryDirectory() as d:
        JSONDataStoreHandler.write_as_files(
            io_test_df, _os.path.join(d, "out-*.json"), lines=True
        )
        got = JSONDataSourceHandler.read_as_pandas(
            _os.path.join(d, "out-0.json"),
            read_kwargs={"head_bytes": "100k", "lines": True},
        )
    assert list(got.columns) == ["a", "b", "c"]
    assert list(got["a"]) == [1, 2, 3]


@pytest.mark.parametrize(
    "fmt, read_dtypes_fn",
    [
        pytest.param(
            "csv",
            lambda d: CSVDataSourceHandler.read_dtypes(_os.path.join(d, "out-0.csv")),
            id="csv",
        ),
        pytest.param(
            "parquet",
            lambda d: ParquetDataSourceHandler.read_dtypes(
                _os.path.join(d, "out-0.parquet")
            ),
            id="parquet",
        ),
        pytest.param(
            "json",
            lambda d: JSONDataSourceHandler.read_dtypes(
                _os.path.join(d, "out-0.json"), read_kwargs={"lines": True}
            ),
            id="json",
        ),
        pytest.param(
            "lance",
            lambda d: LiteFrameReadLanceAdapter._read_dtypes(
                _os.path.join(d, "out.lance")
            ),
            marks=[_requires_lance],
            id="lance",
        ),
    ],
)
@_requires_arrow_dtype
def test_read_dtypes(io_test_df, fmt, read_dtypes_fn):
    write_fn = _SPE_WRITE_FNS[fmt]
    with _tempfile.TemporaryDirectory() as d:
        write_fn(io_test_df, d)
        dtypes = read_dtypes_fn(d)
    keys = [item["key"] for item in dtypes]
    assert keys == ["a", "b", "c"]
    _json.dumps(dtypes)


@_requires_arrow_dtype
def test_csv_usecols_and_dtype_override(io_test_df):
    with _temp_io_path("csv") as (write_path, read_path):
        CSVDataStoreHandler.write_as_files(io_test_df, write_path)
        got = CSVDataSourceHandler.read_as_pandas(read_path, columns=["a", "c"])
    assert list(got.columns) == ["a", "c"]


@_requires_arrow_dtype
def test_parquet_columns_selection(io_test_df):
    with _temp_io_path("parquet") as (write_path, read_path):
        ParquetDataStoreHandler.write_as_files(
            io_test_df, write_path, compression="snappy"
        )
        got = ParquetDataSourceHandler.read_as_pandas(read_path, columns=["b"])
    assert list(got.columns) == ["b"]


@_requires_arrow_dtype
def test_csv_gzipped_round_trip(io_test_df):
    with _temp_io_path("csv.gz") as (write_path, read_path):
        CSVDataStoreHandler.write_as_files(io_test_df, write_path, compression="gzip")
        got = CSVDataSourceHandler.read_as_pandas(
            read_path, read_kwargs={"compression": "infer"}
        )
    assert list(got.columns) == ["a", "b", "c"]
    assert list(got["a"]) == [1, 2, 3]


@_requires_arrow_dtype
def test_csv_uppercase_columns_normalized():
    """Column names from files are lowercased to match LiteFrame semantics."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with _temp_io_path("csv") as (write_path, read_path):
        CSVDataStoreHandler.write_as_files(df, write_path)
        got = CSVDataSourceHandler.read_as_pandas(read_path)
    assert list(got.columns) == ["a", "b"]


@_requires_arrow_dtype
def test_read_csv_nrows_truncation(io_test_df):
    with _temp_io_path("csv") as (write_path, read_path):
        CSVDataStoreHandler.write_as_files(io_test_df, write_path)
        got = CSVDataSourceHandler.read_as_pandas(read_path, nrows=2)
    assert len(got) == 2


@pytest.mark.parametrize(
    "write_fn, read_fn, ext, expected_substring",
    [
        pytest.param(
            lambda df, path: df.to_csv(path, index=False),
            read_csv,
            "csv",
            "CSVDataSourceHandler.read_as_pandas",
            id="csv",
        ),
        pytest.param(
            lambda df, path: LiteFrameToLanceAdapter._write_as_lance(df, path),
            read_lance,
            "lance",
            "LiteFrameReadLanceAdapter._read_as_pandas",
            marks=[_requires_lance],
            id="lance",
        ),
    ],
)
@_requires_arrow_dtype
def test_codegen_end_to_end_read(write_fn, read_fn, ext, expected_substring):
    """Build a read graph and generate+exec SPE code end-to-end."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})
    with _tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, f"src.{ext}")
        write_fn(df, path)
        lf = read_fn(path, _local_test_mode=True)
        dag = lf.build_graph()
        gen = SPECodeGenerator("s1")
        generated = gen.generate(dag)
        assert expected_substring in generated.code
        local_vars = generated.constants.copy()
        local_vars["running"] = True
        exec(generated.code, {}, local_vars)
        result_key = list(generated.output_key_to_variables.values())[0]
        result_df = local_vars[result_key]
    assert list(result_df.columns) == ["a", "b", "c"]
    assert list(result_df["a"]) == [1, 2, 3]


# ---- Lance ----


@_requires_arrow_dtype
@_requires_lance
def test_lance_columns_selection(io_test_df):
    with _tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "out.lance")
        LiteFrameToLanceAdapter._write_as_lance(io_test_df, path)
        got = LiteFrameReadLanceAdapter._read_as_pandas(path, columns=["b"])
    assert list(got.columns) == ["b"]


@_requires_arrow_dtype
@_requires_lance
def test_lance_filters(io_test_df):
    with _tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "out.lance")
        LiteFrameToLanceAdapter._write_as_lance(io_test_df, path)
        got = LiteFrameReadLanceAdapter._read_as_pandas(path, filters="a > 1")
    assert list(got["a"]) == [2, 3]


@_requires_arrow_dtype
@_requires_lance
def test_lance_with_row_address(io_test_df):
    with _tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "out.lance")
        LiteFrameToLanceAdapter._write_as_lance(io_test_df, path)
        got = LiteFrameReadLanceAdapter._read_as_pandas(path, with_row_address=True)
    assert "_rowaddr" in got.columns
    assert len(got) == 3


@_requires_arrow_dtype
@_requires_lance
def test_lance_uppercase_columns_normalized():
    """Mixed-case physical column names are lowercased for LiteFrame."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with _tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "out.lance")
        LiteFrameToLanceAdapter._write_as_lance(df, path)
        got = LiteFrameReadLanceAdapter._read_as_pandas(path, columns=["A", "B"])
    assert list(got.columns) == ["a", "b"]


@_requires_arrow_dtype
def test_csv_read_dtypes_forwards_nrows_and_stops_early(io_test_df):
    """C15: _read_dtypes (nrows=1) forwards nrows=1 to pd.read_csv so the
    schema probe reads one row, and a multi-file read stops opening files once
    the row budget is satisfied."""
    with _tempfile.TemporaryDirectory() as d:
        # Write two files; the dtype probe reads only the first.
        CSVDataStoreHandler.write_as_files(io_test_df, _os.path.join(d, "out-*.csv"))
        paths = sorted(_os.listdir(d))
        assert len(paths) >= 1

        seen_nrows = []
        call_count = {"n": 0}
        real_read_csv = pd.read_csv

        def _spy(*args, **kwargs):
            call_count["n"] += 1
            seen_nrows.append(kwargs.get("nrows"))
            return real_read_csv(*args, **kwargs)

        with mock.patch("pandas.read_csv", side_effect=_spy):
            dtypes = CSVDataSourceHandler.read_dtypes(_os.path.join(d, "out-0.csv"))
        # The dtype probe forwarded nrows=1 to pd.read_csv.
        assert 1 in seen_nrows
        # Schema keys are still resolved correctly from the one-row probe.
        keys = [item["key"] for item in dtypes]
        assert keys == ["a", "b", "c"]

    # Multi-file head-pushdown: a 1-row budget over a 2-file glob opens only
    # the first file and stops.  Write two files *explicitly* so the assertion
    # proves the second is never opened (writing via _write_as_files would
    # produce a single file, making call_count == 1 pass trivially).  Spy on
    # open_file — it receives the path directly; pd.read_csv only sees a file
    # handle and cannot identify which file was read.
    with _tempfile.TemporaryDirectory() as d:
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        p0 = _os.path.join(d, "p-0.csv")
        p1 = _os.path.join(d, "p-1.csv")
        df2.to_csv(p0, index=False)
        df2.to_csv(p1, index=False)
        opened = []

        real_open_file = _ds.open_file

        def _spy2(pth, *args, **kwargs):
            opened.append(str(pth))
            return real_open_file(pth, *args, **kwargs)

        with mock.patch.object(_ds, "open_file", side_effect=_spy2):
            got = CSVDataSourceHandler.read_as_pandas(
                _os.path.join(d, "p-*.csv"), nrows=1
            )
        assert len(got) == 1
        # Exactly one file was opened (budget satisfied by the first file);
        # the other file must never be opened.  _expand_paths order is not
        # guaranteed, so assert on count rather than which file.
        assert len(opened) == 1
        opened_name = _os.path.basename(opened[0])
        assert opened_name in ("p-0.csv", "p-1.csv")
        other = "p-1.csv" if opened_name == "p-0.csv" else "p-0.csv"
        assert not any(other in p for p in opened)
