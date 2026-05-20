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

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from maxframe.failures import FailureInfo, _load_table_meta, get_failure_info
from maxframe.io.odpsio.arrow import pandas_to_arrow


def _attach_failure_info(exc: Exception, info: dict) -> Exception:
    exc._failure_info = info
    return exc


def test_failure_info_fetch_and_repr():
    info = FailureInfo(
        substep_id="substep-1",
        operator_name="DataFrameApply",
        error_message="boom",
        table="proj.tmp_tbl",
    )
    assert "proj.tmp_tbl" in repr(info)
    assert FailureInfo().fetch_data() is None

    expected_df = pd.DataFrame({"a": [1]})
    expected_arrow, expected_meta = pandas_to_arrow(expected_df)
    reader = MagicMock()
    reader.__enter__.return_value = reader
    reader.__exit__.return_value = False
    reader.read_all.return_value = expected_arrow

    table_io_instance = MagicMock()
    table_io_instance.open_reader.return_value = reader
    odps_entry = object()

    info = FailureInfo(table="proj.tbl", table_meta=expected_meta)
    sync_cm = MagicMock()
    sync_cm.__enter__.return_value = None
    sync_cm.__exit__.return_value = False
    with patch("maxframe.failures.ODPSTableIO", return_value=table_io_instance) as mio:
        with patch("maxframe.failures.sync_pyodps_options", return_value=sync_cm):
            out = info.fetch_data(odps_entry=odps_entry)
    mio.assert_called_once_with(odps_entry)
    table_io_instance.open_reader.assert_called_once_with("proj.tbl")
    pd.testing.assert_frame_equal(out, expected_df)
    assert info.fetch_data(odps_entry=odps_entry) is out

    info = FailureInfo(table="proj.tbl")
    with patch(
        "maxframe.failures.ODPS.from_environments", side_effect=RuntimeError("no creds")
    ):
        with pytest.raises(ValueError, match="pass an odps_entry") as exc_info:
            info.fetch_data()
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_get_failure_info():
    direct = _attach_failure_info(
        RuntimeError("outer"),
        {"substep_id": "s1", "operator_name": "DataFrameApply", "table": "p.t"},
    )
    assert get_failure_info(direct).substep_id == "s1"

    cause = _attach_failure_info(
        ValueError("cause"),
        {"substep_id": "s2", "operator_name": "DataFrameApply", "table": "p.t2"},
    )
    outer = RuntimeError("outer")
    outer.__cause__ = cause
    assert get_failure_info(outer).substep_id == "s2"

    failure_df = pd.DataFrame({"a": [1]}, index=[5])
    failure_arrow, failure_meta = pandas_to_arrow(failure_df)
    reader = MagicMock()
    reader.__enter__.return_value = reader
    reader.__exit__.return_value = False
    reader.read_all.return_value = failure_arrow
    table_io_instance = MagicMock()
    table_io_instance.open_reader.return_value = reader
    direct = _attach_failure_info(
        RuntimeError("outer"),
        {
            "table": "p.t3",
            "table_meta": failure_meta.to_json(),
            "substep_id": "s3",
            "operator_name": "DataFrameApply",
        },
    )
    sync_cm = MagicMock()
    sync_cm.__enter__.return_value = None
    sync_cm.__exit__.return_value = False
    with patch("maxframe.failures.ODPSTableIO", return_value=table_io_instance):
        with patch("maxframe.failures.sync_pyodps_options", return_value=sync_cm):
            info = get_failure_info(direct, odps_entry=object())
    assert info.substep_id == "s3"
    assert info.operator_name == "DataFrameApply"
    assert "substep_id='s3'" in repr(info)
    assert "operator_name='DataFrameApply'" in repr(info)
    pd.testing.assert_frame_equal(info.fetch_data(odps_entry=object()), failure_df)

    info = get_failure_info(
        _attach_failure_info(
            RuntimeError("outer"),
            {"table": "p.t4", "table_meta": failure_meta.to_json()},
        )
    )
    assert info.substep_id is None
    assert info.operator_name is None

    assert get_failure_info(ValueError("none")) is None


def test_load_table_meta_warns_on_invalid_input():
    with pytest.warns(RuntimeWarning, match="Failed to deserialize DataFrameTableMeta"):
        assert _load_table_meta({"bad": "data"}) is None

    with pytest.warns(
        RuntimeWarning, match="Failed to construct DataFrameTableMeta from dict"
    ):
        assert (
            _load_table_meta(
                {
                    "type": "bad-type",
                    "pd_column_dtypes": pd.Series([1], index=["a"]),
                }
            )
            is None
        )
