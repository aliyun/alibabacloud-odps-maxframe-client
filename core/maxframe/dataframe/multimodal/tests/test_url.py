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

import numpy as np
import pandas as pd
import pytest

from .... import opcodes
from ....core import OutputType
from ...core import SERIES_TYPE
from ...datasource.series import from_pandas as from_pandas_series
from ..url import SeriesUrlMethods, UrlAccessor, url_download


def test_url_download():
    s = pd.Series(
        [
            "oss://oss-maxframe-test.aliyuncs.com/maxframe-test/image/file1.jpg",
            "oss://oss-maxframe-test.aliyuncs.com/maxframe-test/image/file2.png",
            "oss://oss-maxframe-test.aliyuncs.com/maxframe-test/image/file3.bmp",
        ],
        name="urls",
    )
    series = from_pandas_series(s, chunk_size=2)

    # Test via function without storage_options
    r = url_download(series)
    assert isinstance(r, SERIES_TYPE)
    assert r.dtype == np.dtype(object)
    assert r.shape == s.shape
    assert r.name == s.name
    assert r.op._op_type_ == opcodes.SERIES_URL_METHODS
    assert r.op.method == "download"
    assert r.op.output_types[0] == OutputType.series
    assert r.op.storage_options is None

    # Test via function with storage_options
    opts = {"role_arn": "acs:ram::123456:role/test-role"}
    r = url_download(series, storage_options=opts)
    assert isinstance(r, SERIES_TYPE)
    assert r.dtype == np.dtype(object)
    assert r.op.storage_options == opts

    # Test via accessor without storage_options
    r = series.url.download()
    assert isinstance(r, SERIES_TYPE)
    assert r.dtype == np.dtype(object)
    assert r.shape == s.shape
    assert r.op._op_type_ == opcodes.SERIES_URL_METHODS
    assert r.op.method == "download"
    assert r.op.storage_options is None

    # Test via accessor with storage_options
    r = series.url.download(storage_options=opts)
    assert isinstance(r, SERIES_TYPE)
    assert r.op.storage_options == opts


def test_url_download_operator():
    op = SeriesUrlMethods(method="download")
    assert op.output_types == [OutputType.series]
    assert op.method == "download"

    opts = {"key": "value", "secret": "abc"}
    op = SeriesUrlMethods(method="download", storage_options=opts)
    assert op.storage_options == opts


def test_url_download_preserves_index():
    s = pd.Series(
        [
            "oss://oss-maxframe-test.aliyuncs.com/maxframe-test/image/a.jpg",
            "oss://oss-maxframe-test.aliyuncs.com/maxframe-test/image/b.jpg",
        ],
        index=pd.Index(["row1", "row2"]),
        name="file_urls",
    )
    series = from_pandas_series(s, chunk_size=2)

    r = url_download(series)
    assert r.index_value.key == series.index_value.key
    assert r.name == "file_urls"


def test_url_download_storage_options_types():
    s = pd.Series(
        ["oss://oss-maxframe-test.aliyuncs.com/maxframe-test/image/file.jpg"],
        name="urls",
    )
    series = from_pandas_series(s, chunk_size=1)

    # Empty dict
    r = url_download(series, storage_options={})
    assert r.op.storage_options == {}

    # Dict with multiple keys
    opts = {
        "role_arn": "acs:ram::123:role/test",
        "endpoint": "oss-maxframe-test.aliyuncs.com",
        "access_key_id": "test_id",
        "access_key_secret": "test_secret",
    }
    r = url_download(series, storage_options=opts)
    assert r.op.storage_options == opts
    assert r.op.storage_options["role_arn"] == "acs:ram::123:role/test"


def test_url_accessor_type_check():
    with pytest.raises(AttributeError, match="Can only use .url accessor with Series"):
        UrlAccessor("not a series")

    with pytest.raises(AttributeError, match="Can only use .url accessor with Series"):
        UrlAccessor(pd.Series([1, 2, 3]))


def test_url_download_chain_to_image():
    s = pd.Series(
        [
            "oss://oss-maxframe-test.aliyuncs.com/maxframe-test/image/img1.jpg",
            "oss://oss-maxframe-test.aliyuncs.com/maxframe-test/image/img2.png",
        ],
        name="image_urls",
    )
    series = from_pandas_series(s, chunk_size=2)

    # Chain: url.download() -> image.decode()
    downloaded = series.url.download(
        storage_options={"role_arn": "acs:ram::123:role/test"}
    )
    assert downloaded.op._op_type_ == opcodes.SERIES_URL_METHODS
    assert downloaded.op.method == "download"
    assert downloaded.dtype == np.dtype(object)

    decoded = downloaded.image.decode()
    assert decoded.op._op_type_ == opcodes.SERIES_IMAGE_METHODS
    assert decoded.op.method == "decode"
    assert decoded.dtype == np.dtype(object)

    # Chain: url.download() -> image.decode() -> image.width
    width = decoded.image.width
    assert width.op._op_type_ == opcodes.SERIES_IMAGE_METHODS
    assert width.op.method == "width"
    assert width.dtype == pd.Int64Dtype()
