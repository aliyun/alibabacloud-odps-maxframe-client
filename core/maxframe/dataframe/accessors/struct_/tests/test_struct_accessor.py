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

import pandas as pd
import pyarrow as pa
import pytest

from ..... import dataframe as md
from .....utils import ARROW_DTYPE_NOT_SUPPORTED
from ..core import SeriesStructMethod

pytestmark = pytest.mark.skipif(
    ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported"
)


@pytest.fixture
def series():
    version_type = pa.struct(
        [
            ("major", pa.int64()),
            ("minor", pa.int64()),
        ]
    )
    return md.Series(
        [
            {"version": {"major": 1, "minor": 5}, "project": "pandas"},
            {"version": {"major": 2, "minor": 1}, "project": "pandas"},
            {"version": {"major": 1, "minor": 26}, "project": "numpy"},
        ],
        dtype=pd.ArrowDtype(
            pa.struct([("version", version_type), ("project", pa.string())])
        ),
    )


def test_dtypes(series):
    pd.testing.assert_series_equal(
        series.struct.dtypes,
        pd.Series(
            [
                pd.ArrowDtype(
                    pa.struct([("major", pa.int64()), ("minor", pa.int64())])
                ),
                pd.ArrowDtype(pa.string()),
            ],
            index=["version", "project"],
        ),
    )


def test_field(series):
    version_type = pa.struct(
        [
            ("major", pa.int64()),
            ("minor", pa.int64()),
        ]
    )

    s1 = series.struct.field("version")
    assert isinstance(s1, md.Series)
    assert s1.name == "version"
    assert s1.dtype == pd.ArrowDtype(version_type)
    assert s1.shape == (3,)
    assert s1.index_value == series.index_value
    op = s1.op
    assert isinstance(op, SeriesStructMethod)
    assert op.method == "field"
    assert op.method_kwargs["name_or_index"] == "version"

    s2 = series.struct.field(["version", "major"])
    assert isinstance(s1, md.Series)
    assert s2.name == "major"
    assert s2.dtype == pd.ArrowDtype(pa.int64())
    assert s2.shape == (3,)
    assert s2.index_value == series.index_value
    op = s2.op
    assert isinstance(op, SeriesStructMethod)
    assert op.method == "field"
    assert op.method_kwargs["name_or_index"] == ["version", "major"]
