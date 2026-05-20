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

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from maxframe.dataframe.datasource.series import from_pandas as from_pandas_series
from maxframe.dataframe.multimodal.audio import (
    AudioAccessor,
    AudioObject,
    SeriesAudioMethods,
    audio_decode,
    audio_property,
    detect_language,
    transcribe,
    vad_detect,
)
from maxframe.dataframe.multimodal.audio.objects import TranscribeResult, VadSegment
from maxframe.utils import is_arrow_dtype_supported


def test_audio_object():
    audio = AudioObject(
        data=b"fake_audio_data",
        sample_rate=16000,
        duration=1.5,
        channels=1,
        format="flac",
    )
    assert audio.data == b"fake_audio_data"
    assert audio.sample_rate == 16000
    assert audio.duration == 1.5
    assert audio.channels == 1
    assert audio.format == "flac"
    assert audio.size == len(b"fake_audio_data")

    # Test default values
    audio2 = AudioObject(data=b"data")
    assert audio2.sample_rate is None
    assert audio2.duration is None
    assert audio2.channels is None
    assert audio2.format is None
    assert audio2.size == 4

    # Test empty / None data
    audio3 = AudioObject(data=None)
    assert audio3.size == 0

    audio4 = AudioObject(data=b"")
    assert audio4.size == 0


def test_audio_operator_contract():
    series_data = pd.Series(
        [b"audio_1", b"audio_2", b"audio_3"],
        index=pd.Index(["row1", "row2", "row3"]),
        name="audio",
    )
    series = from_pandas_series(series_data, chunk_size=2)

    with pytest.raises(
        AttributeError, match="Can only use .audio accessor with Series"
    ):
        AudioAccessor("not a series")
    with pytest.raises(
        AttributeError, match="Can only use .audio accessor with Series"
    ):
        AudioAccessor(pd.Series([1, 2, 3]))

    decoded = audio_decode(series, target_sample_rate=8000, max_duration_sec=10.0)
    assert isinstance(decoded, SERIES_TYPE)
    assert decoded.dtype == np.dtype(object)
    assert decoded.shape == series_data.shape
    assert decoded.name == series.name
    assert decoded.index_value.key == series.index_value.key
    assert decoded.op._op_type_ == opcodes.SERIES_AUDIO_METHODS
    assert decoded.op.method == "decode"
    assert decoded.op.target_sample_rate == 8000
    assert decoded.op.max_duration_sec == 10.0
    assert decoded.op.output_types[0] == OutputType.series
    assert decoded.op.has_custom_code() is False

    direct_decoded = SeriesAudioMethods(method="decode")(series)
    assert isinstance(direct_decoded, SERIES_TYPE)
    assert direct_decoded.dtype == np.dtype(object)

    accessor_decoded = series.audio.decode(target_sample_rate=22050)
    assert isinstance(accessor_decoded, SERIES_TYPE)
    assert accessor_decoded.op.method == "decode"
    assert accessor_decoded.op.target_sample_rate == 22050

    expected_dtypes = {
        "sample_rate": pd.Int64Dtype(),
        "channels": pd.Int64Dtype(),
        "size": pd.Int64Dtype(),
        "duration": np.dtype("float64"),
        "format": np.dtype(object),
    }
    for prop, expected_dtype in expected_dtypes.items():
        result = audio_property(series, prop)
        assert isinstance(result, SERIES_TYPE)
        assert result.dtype == expected_dtype
        assert result.shape == series_data.shape
        assert result.name == series.name
        assert result.index_value.key == series.index_value.key
        assert result.op._op_type_ == opcodes.SERIES_AUDIO_METHODS
        assert result.op.method == prop
        assert result.op.output_types[0] == OutputType.series
        assert result.op.has_custom_code() is False

        accessor_result = getattr(accessor_decoded.audio, prop)
        assert isinstance(accessor_result, SERIES_TYPE)
        assert accessor_result.dtype == expected_dtype
        assert accessor_result.op.method == prop

        prop_op = SeriesAudioMethods(method=prop)
        assert prop_op.method == prop
        assert prop_op.output_types == [OutputType.series]
        assert prop_op.has_custom_code() is False


def test_asr_operator_contract(monkeypatch):
    from maxframe.dataframe.multimodal.audio import core as audio_core

    s = pd.Series(
        [b"audio_1", b"audio_2"],
        index=pd.Index(["row1", "row2"]),
        name="audio",
    )
    series = from_pandas_series(s, chunk_size=2)

    detect_result = detect_language(series, gu=1, gu_quota="mf_gpu_quota")
    assert isinstance(detect_result, SERIES_TYPE)
    assert detect_result.dtype == np.dtype(object)
    assert detect_result.shape == s.shape
    assert detect_result.name == series.name
    assert detect_result.index_value.key == series.index_value.key
    assert detect_result.op._op_type_ == opcodes.SERIES_AUDIO_METHODS
    assert detect_result.op.method == "detect_language"
    assert detect_result.op.output_types[0] == OutputType.series
    assert detect_result.op.gu == 1
    assert detect_result.op.gu_quota == ["mf_gpu_quota"]
    assert detect_result.op.gpu is True
    assert detect_result.op.has_custom_code() is True

    detect_accessor = series.audio.detect_language(max_duration_sec=15.0)
    assert detect_accessor.op.method == "detect_language"
    assert detect_accessor.op.max_duration_sec == 15.0

    transcribed = transcribe(series, language="zh", gu=2, gu_quota=["q1", "q2"])
    expected_dtypes = TranscribeResult.output_dtypes()
    assert isinstance(transcribed, DATAFRAME_TYPE)
    assert list(transcribed.dtypes.index) == list(expected_dtypes.index)
    assert transcribed.dtypes["text"] == np.dtype("object")
    assert transcribed.dtypes["language"] == np.dtype("object")
    assert transcribed.dtypes["language_probability"] == np.dtype("float64")
    assert transcribed.dtypes["segments"] == np.dtype("object")
    assert transcribed.shape == (s.shape[0], len(expected_dtypes))
    assert transcribed.index_value.key == series.index_value.key
    assert transcribed.op._op_type_ == opcodes.SERIES_AUDIO_METHODS
    assert transcribed.op.method == "transcribe"
    assert transcribed.op.language == "zh"
    assert transcribed.op.gu == 2
    assert transcribed.op.gu_quota == ["q1", "q2"]
    assert transcribed.op.gpu is True
    assert transcribed.op.has_custom_code() is True
    assert transcribed.op.output_types[0] == OutputType.dataframe

    transcribe_accessor = series.audio.transcribe(language="en")
    assert transcribe_accessor.op.language == "en"
    assert transcribe_accessor.op.method == "transcribe"
    assert transcribe_accessor.index_value.key == series.index_value.key

    vad_result = vad_detect(series, threshold=0.7, gu=1, gu_quota="vad_quota")
    assert isinstance(vad_result, SERIES_TYPE)
    assert vad_result.dtype == (
        VadSegment.dtype() if is_arrow_dtype_supported() else np.dtype(object)
    )
    assert vad_result.shape == s.shape
    assert vad_result.name == series.name
    assert vad_result.index_value.key == series.index_value.key
    assert vad_result.op._op_type_ == opcodes.SERIES_AUDIO_METHODS
    assert vad_result.op.method == "vad_detect"
    assert vad_result.op.threshold == 0.7
    assert vad_result.op.gu == 1
    assert vad_result.op.gu_quota == ["vad_quota"]
    assert vad_result.op.gpu is True
    assert vad_result.op.has_custom_code() is True

    vad_accessor = series.audio.vad_detect(threshold=0.6)
    assert vad_accessor.op.threshold == 0.6
    assert vad_accessor.op.method == "vad_detect"

    op = SeriesAudioMethods(method="detect_language")
    assert not hasattr(op, "model_name")
    assert not hasattr(op, "model_file")

    def _unsupported_dtype():
        raise AssertionError("VadSegment.dtype should not be called")

    monkeypatch.setattr(
        audio_core, "is_arrow_dtype_supported", lambda: False, raising=False
    )
    monkeypatch.setattr(VadSegment, "dtype", staticmethod(_unsupported_dtype))

    fallback = vad_detect(series)
    assert isinstance(fallback, SERIES_TYPE)
    assert fallback.dtype == np.dtype(object)
    assert fallback.index_value.key == series.index_value.key
    assert fallback.name == "audio"
