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

from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.dataframe.multimodal.audio.objects import TranscribeResult, VadSegment
from maxframe.dataframe.operators import DataFrameOperator, DataFrameOperatorMixin
from maxframe.dataframe.utils import parse_index
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    FieldTypes,
    Float64Field,
    Int64Field,
    ListField,
    StringField,
)
from maxframe.utils import is_arrow_dtype_supported

if TYPE_CHECKING:
    from ...core import DataFrame, Series

_INT_METHODS = ("sample_rate", "channels", "size")
_FLOAT_METHODS = ("duration",)
_STR_METHODS = ("format",)
_ASR_METHODS = ("detect_language", "transcribe", "vad_detect")
_VAD_DETECT_METHOD = "vad_detect"


class SeriesAudioMethods(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.SERIES_AUDIO_METHODS

    method = StringField("method", default=None)

    # Params for method == "decode"
    target_sample_rate = Int64Field("target_sample_rate", default=16000)
    max_duration_sec = Float64Field("max_duration_sec", default=None)

    # Params for method == "transcribe"
    language = StringField("language", default=None)

    # Params for method == "vad_detect"
    threshold = Float64Field("threshold", default=0.5)
    min_speech_duration_ms = Int64Field("min_speech_duration_ms", default=250)
    min_silence_duration_ms = Int64Field("min_silence_duration_ms", default=100)

    # Resource configuration (used by ASR methods)
    cpu = Int64Field("cpu", default=None)
    memory = AnyField("memory", default=None)
    gu = Int64Field("gu", default=None)
    gu_quota = ListField("gu_quota", FieldTypes.string, default=None)
    gpu = BoolField("gpu", default=False)

    def __init__(self, output_types=None, **kw):
        if output_types is None:
            output_types = (
                [OutputType.dataframe]
                if kw.get("method") == "transcribe"
                else [OutputType.series]
            )
        kw["_output_types"] = kw.get("_output_types") or output_types
        super().__init__(**kw)

    def has_custom_code(self) -> bool:
        # ASR methods require rund container for fs_mount (whisper model)
        return self.method in _ASR_METHODS

    def __call__(self, series: "Series") -> "Union[Series, DataFrame]":
        if self.method == "transcribe":
            _dtypes = TranscribeResult.output_dtypes()
            return self.new_dataframe(
                [series],
                shape=(series.shape[0], len(_dtypes)),
                dtypes=_dtypes,
                index_value=series.index_value,
                columns_value=parse_index(_dtypes.index, store_data=True),
            )
        dtype = self._resolve_dtype()
        return self.new_series(
            [series],
            shape=series.shape,
            dtype=dtype,
            index_value=series.index_value,
            name=series.name,
        )

    def _resolve_dtype(self):
        if self.method in _INT_METHODS:
            return pd.Int64Dtype()
        elif self.method in _FLOAT_METHODS:
            return np.dtype("float64")
        elif self.method in _STR_METHODS:
            return np.dtype(object)
        elif self.method == _VAD_DETECT_METHOD:
            return (
                VadSegment.dtype() if is_arrow_dtype_supported() else np.dtype(object)
            )
        else:
            # decode, detect_language all return object
            return np.dtype(object)


def audio_decode(
    series: "Series",
    target_sample_rate: int = 16000,
    max_duration_sec: Optional[float] = None,
) -> "Series":
    op = SeriesAudioMethods(
        method="decode",
        target_sample_rate=target_sample_rate,
        max_duration_sec=max_duration_sec,
    )
    return op(series)


def audio_property(series: "Series", prop_name: str) -> "Series":
    op = SeriesAudioMethods(method=prop_name)
    return op(series)


def detect_language(
    series: "Series",
    max_duration_sec: float = 30.0,
    cpu: Optional[int] = None,
    memory: Optional[Union[str, int]] = None,
    gu: Optional[int] = None,
    gu_quota: Optional[Union[str, List[str]]] = None,
) -> "Series":
    """
    Detect spoken language in audio.

    Parameters
    ----------
    series : Series
        Series containing audio bytes.
    max_duration_sec : float, default 30.0
        Maximum duration to analyze (first 30s is usually enough).
    cpu : int, optional
        Number of CPUs for the rund container.
    memory : str or int, optional
        Memory for the rund container. If int, unit is GB (e.g., 8 means 8GB).
        If str, specify with unit (e.g., "8GiB", "8192MiB").
    gu : int, optional
        Number of GPUs for the rund container.
    gu_quota : str or list of str, optional
        GPU quota nicknames. The order is the priority of the usage.

    Returns
    -------
    Series
        Language codes (ISO 639-1): "en", "zh", "ja", etc.

    Examples
    --------
    >>> df["language"] = df["audio"].audio.detect_language()
    >>> df["language"] = df["audio"].audio.detect_language(cpu=4, memory="16GiB")
    >>> df["language"] = df["audio"].audio.detect_language(gu=1, gu_quota="mf_gpu_quota")
    >>> df[df["language"] == "en"]  # Filter English audio
    """
    if isinstance(gu_quota, str):
        gu_quota = [gu_quota]
    op = SeriesAudioMethods(
        method="detect_language",
        max_duration_sec=max_duration_sec,
        cpu=cpu,
        memory=memory,
        gu=gu,
        gu_quota=gu_quota,
        gpu=gu is not None and gu > 0,
    )
    return op(series)


def transcribe(
    series: "Series",
    language: Optional[str] = None,
    max_duration_sec: Optional[float] = None,
    cpu: Optional[int] = None,
    memory: Optional[Union[str, int]] = None,
    gu: Optional[int] = None,
    gu_quota: Optional[Union[str, List[str]]] = None,
) -> "Series":
    """
    Transcribe audio to text.

    Parameters
    ----------
    series : Series
        Series containing audio bytes.
    language : str, optional
        Target language ("en", "zh", "ja", etc.). Auto-detect if None.
    max_duration_sec : float, optional
        Maximum duration to transcribe. Full audio if None.
    cpu : int, optional
        Number of CPUs for the rund container.
    memory : str or int, optional
        Memory for the rund container. If int, unit is GB (e.g., 8 means 8GB).
        If str, specify with unit (e.g., "8GiB", "8192MiB").
    gu : int, optional
        Number of GPUs for the rund container.
    gu_quota : str or list of str, optional
        GPU quota nicknames. The order is the priority of the usage.

    Returns
    -------
    DataFrame
        DataFrame with columns:

        - ``text`` (str): full transcribed text.
        - ``language`` (str): detected language code, e.g. ``"zh"``, ``"en"``.
        - ``language_probability`` (float): confidence of the detected language (0–1).
        - ``segments`` (str): JSON string of per-segment details, each with
          ``start``, ``end``, and ``text`` fields.

    Examples
    --------
    >>> result = df["audio"].audio.transcribe()
    >>> df["text"] = result["text"]
    >>> df["lang"] = result["language"]
    >>> df["transcript"] = df["audio"].audio.transcribe(language="zh", cpu=4, memory="16GiB")["text"]
    >>> df["transcript"] = df["audio"].audio.transcribe(gu=1, gu_quota="mf_gpu_quota")["text"]
    """
    if isinstance(gu_quota, str):
        gu_quota = [gu_quota]
    op = SeriesAudioMethods(
        method="transcribe",
        language=language,
        max_duration_sec=max_duration_sec,
        cpu=cpu,
        memory=memory,
        gu=gu,
        gu_quota=gu_quota,
        gpu=gu is not None and gu > 0,
    )
    return op(series)


def vad_detect(
    series: "Series",
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    cpu: Optional[int] = None,
    memory: Optional[Union[str, int]] = None,
    gu: Optional[int] = None,
    gu_quota: Optional[Union[str, List[str]]] = None,
) -> "Series":
    """
    Detect voice activity segments in audio.

    Parameters
    ----------
    series : Series
        Series containing audio bytes.
    threshold : float, default 0.5
        VAD threshold (0.0 to 1.0).
    min_speech_duration_ms : int, default 250
        Minimum speech duration in milliseconds.
    min_silence_duration_ms : int, default 100
        Minimum silence duration in milliseconds.
    cpu : int, optional
        Number of CPUs for the rund container.
    memory : str or int, optional
        Memory for the rund container. If int, unit is GB (e.g., 8 means 8GB).
        If str, specify with unit (e.g., "8GiB", "8192MiB").
    gu : int, optional
        Number of GPUs for the rund container.
    gu_quota : str or list of str, optional
        GPU quota nicknames. The order is the priority of the usage.

    Returns
    -------
    Series
        JSON strings containing VAD segments, e.g.
        ``[{"start": 0.12, "end": 1.34}]``.

    Examples
    --------
    >>> df["vad"] = df["audio"].audio.vad_detect()
    >>> df["vad"] = df["audio"].audio.vad_detect(gu=1, gu_quota="mf_gpu_quota")
    """
    if isinstance(gu_quota, str):
        gu_quota = [gu_quota]
    op = SeriesAudioMethods(
        method="vad_detect",
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        cpu=cpu,
        memory=memory,
        gu=gu,
        gu_quota=gu_quota,
        gpu=gu is not None and gu > 0,
    )
    return op(series)
