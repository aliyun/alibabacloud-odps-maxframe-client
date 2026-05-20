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

from maxframe.core import BaseMaxFrameAccessor
from maxframe.dataframe.core import SERIES_TYPE
from maxframe.dataframe.multimodal.audio.core import (
    audio_decode,
    audio_property,
    detect_language,
    transcribe,
    vad_detect,
)

if TYPE_CHECKING:
    from maxframe.dataframe.core import Series


class AudioAccessor(BaseMaxFrameAccessor):
    """
    Examples
    --------
    >>> df["audio"] = df["path"].url.download()
    >>> df["language"] = df["audio"].audio.detect_language()
    >>> df["transcript"] = df["audio"].audio.transcribe(language="zh")
    """

    obj: "Series"
    _api_count: int = 9  # decode, detect_language, transcribe, vad_detect,
    #                       sample_rate, duration, channels, size, format

    def __init__(self, series: "Series"):
        if not isinstance(series, SERIES_TYPE):
            raise AttributeError("Can only use .audio accessor with Series")
        super().__init__(series)

    def decode(
        self,
        target_sample_rate: int = 16000,
        max_duration_sec: Optional[float] = None,
    ) -> "Series":
        return audio_decode(
            self.obj,
            target_sample_rate=target_sample_rate,
            max_duration_sec=max_duration_sec,
        )

    def detect_language(
        self,
        max_duration_sec: float = 30.0,
        cpu: Optional[int] = None,
        memory: Optional[Union[str, int]] = None,
        gu: Optional[int] = None,
        gu_quota: Optional[Union[str, List[str]]] = None,
    ) -> "Series":
        return detect_language(
            self.obj,
            max_duration_sec=max_duration_sec,
            cpu=cpu,
            memory=memory,
            gu=gu,
            gu_quota=gu_quota,
        )

    def transcribe(
        self,
        language: Optional[str] = None,
        max_duration_sec: Optional[float] = None,
        cpu: Optional[int] = None,
        memory: Optional[Union[str, int]] = None,
        gu: Optional[int] = None,
        gu_quota: Optional[Union[str, List[str]]] = None,
    ) -> "Series":
        return transcribe(
            self.obj,
            language=language,
            max_duration_sec=max_duration_sec,
            cpu=cpu,
            memory=memory,
            gu=gu,
            gu_quota=gu_quota,
        )

    def vad_detect(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        cpu: Optional[int] = None,
        memory: Optional[Union[str, int]] = None,
        gu: Optional[int] = None,
        gu_quota: Optional[Union[str, List[str]]] = None,
    ) -> "Series":
        return vad_detect(
            self.obj,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            cpu=cpu,
            memory=memory,
            gu=gu,
            gu_quota=gu_quota,
        )

    @property
    def sample_rate(self) -> "Series":
        return audio_property(self.obj, "sample_rate")

    @property
    def duration(self) -> "Series":
        return audio_property(self.obj, "duration")

    @property
    def channels(self) -> "Series":
        return audio_property(self.obj, "channels")

    @property
    def size(self) -> "Series":
        return audio_property(self.obj, "size")

    @property
    def format(self) -> "Series":
        return audio_property(self.obj, "format")
