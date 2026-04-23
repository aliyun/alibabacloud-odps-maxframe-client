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
from ..image import (
    ImageAccessor,
    ImageObject,
    SeriesImageMethods,
    image_decode,
    image_property,
)


def test_image_decode():
    s = pd.Series(["path1.jpg", "path2.png", "path3.bmp"], name="images")
    series = from_pandas_series(s, chunk_size=2)

    # Test via function
    r = image_decode(series)
    assert isinstance(r, SERIES_TYPE)
    assert r.dtype == np.dtype(object)
    assert r.shape == s.shape
    assert r.name == s.name
    assert r.op._op_type_ == opcodes.SERIES_IMAGE_METHODS
    assert r.op.method == "decode"
    assert r.op.output_types[0] == OutputType.series

    # Test operator directly
    op = SeriesImageMethods(method="decode")
    assert op.output_types == [OutputType.series]
    result = op(series)
    assert isinstance(result, SERIES_TYPE)
    assert result.dtype == np.dtype(object)

    # Test via accessor
    r = series.image.decode()
    assert isinstance(r, SERIES_TYPE)
    assert r.dtype == np.dtype(object)
    assert r.shape == s.shape
    assert r.op._op_type_ == opcodes.SERIES_IMAGE_METHODS
    assert r.op.method == "decode"


@pytest.mark.parametrize("prop", ["width", "height", "size"])
def test_image_int_property(prop):
    s = pd.Series(["img_data_1", "img_data_2"], name="decoded")
    series = from_pandas_series(s, chunk_size=2)

    # Test via function
    r = image_property(series, prop)
    assert isinstance(r, SERIES_TYPE)
    assert r.dtype == pd.Int64Dtype()
    assert r.shape == s.shape
    assert r.name == s.name
    assert r.op._op_type_ == opcodes.SERIES_IMAGE_METHODS
    assert r.op.method == prop
    assert r.op.output_types[0] == OutputType.series

    # Test via accessor
    r = getattr(series.image, prop)
    assert isinstance(r, SERIES_TYPE)
    assert r.dtype == pd.Int64Dtype()
    assert r.op.method == prop


@pytest.mark.parametrize("prop", ["mode", "format"])
def test_image_str_property(prop):
    s = pd.Series(["img_data_1", "img_data_2"], name="decoded")
    series = from_pandas_series(s, chunk_size=2)

    # Test via function
    r = image_property(series, prop)
    assert isinstance(r, SERIES_TYPE)
    assert r.dtype == np.dtype(object)
    assert r.shape == s.shape
    assert r.name == s.name
    assert r.op._op_type_ == opcodes.SERIES_IMAGE_METHODS
    assert r.op.method == prop
    assert r.op.output_types[0] == OutputType.series

    # Test via accessor
    r = getattr(series.image, prop)
    assert isinstance(r, SERIES_TYPE)
    assert r.dtype == np.dtype(object)
    assert r.op.method == prop


def test_image_property_operator():
    # int64 properties
    for prop in ("width", "height", "size"):
        op = SeriesImageMethods(method=prop)
        assert op.method == prop
        assert op.output_types == [OutputType.series]

    # object properties
    for prop in ("mode", "format"):
        op = SeriesImageMethods(method=prop)
        assert op.method == prop


def test_image_accessor_type_check():
    with pytest.raises(
        AttributeError, match="Can only use .image accessor with Series"
    ):
        ImageAccessor("not a series")

    with pytest.raises(
        AttributeError, match="Can only use .image accessor with Series"
    ):
        ImageAccessor(pd.Series([1, 2, 3]))


def test_image_decode_preserves_index():
    s = pd.Series(
        ["path1.jpg", "path2.png"],
        index=pd.Index(["a", "b"]),
        name="imgs",
    )
    series = from_pandas_series(s, chunk_size=2)

    r = image_decode(series)
    assert r.index_value.key == series.index_value.key
    assert r.name == "imgs"


def test_image_property_preserves_index():
    s = pd.Series(
        ["data1", "data2"],
        index=pd.Index(["x", "y"]),
        name="images",
    )
    series = from_pandas_series(s, chunk_size=2)

    for prop in ("width", "height", "size", "mode", "format"):
        r = image_property(series, prop)
        assert r.index_value.key == series.index_value.key
        assert r.name == "images"


def test_image_chain_operations():
    s = pd.Series(["path1.jpg", "path2.png", "path3.bmp"], name="paths")
    series = from_pandas_series(s, chunk_size=2)

    decoded = series.image.decode()
    assert decoded.op._op_type_ == opcodes.SERIES_IMAGE_METHODS
    assert decoded.op.method == "decode"

    width = decoded.image.width
    assert width.op._op_type_ == opcodes.SERIES_IMAGE_METHODS
    assert width.op.method == "width"
    assert width.dtype == pd.Int64Dtype()

    height = decoded.image.height
    assert height.op._op_type_ == opcodes.SERIES_IMAGE_METHODS
    assert height.op.method == "height"
    assert height.dtype == pd.Int64Dtype()


def test_image_object():
    img = ImageObject(
        data=b"fake_image_data", width=100, height=200, mode="RGB", format="JPEG"
    )
    assert img.data == b"fake_image_data"
    assert img.width == 100
    assert img.height == 200
    assert img.mode == "RGB"
    assert img.format == "JPEG"
    assert img.size == len(b"fake_image_data")

    # Test default values
    img2 = ImageObject(data=b"data")
    assert img2.width is None
    assert img2.height is None
    assert img2.mode is None
    assert img2.format is None
    assert img2.size == 4

    # Test empty data
    img3 = ImageObject(data=None)
    assert img3.size == 0

    img4 = ImageObject(data=b"")
    assert img4.size == 0
