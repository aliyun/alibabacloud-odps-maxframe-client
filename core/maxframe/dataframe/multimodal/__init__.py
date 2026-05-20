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

from maxframe.dataframe.multimodal.audio import (
    AudioAccessor,
    AudioObject,
    SeriesAudioMethods,
    TranscribeResult,
    VadSegment,
    audio_decode,
    audio_property,
    detect_language,
    transcribe,
    vad_detect,
)
from maxframe.dataframe.multimodal.image import (
    ImageAccessor,
    ImageObject,
    SeriesImageMethods,
    image_decode,
    image_property,
)
from maxframe.dataframe.multimodal.url import (
    SeriesUrlMethods,
    UrlAccessor,
    url_download,
)


def _install():
    from maxframe.core import CachedAccessor
    from maxframe.dataframe.core import SERIES_TYPE

    # Register accessors to Series types
    if UrlAccessor._api_count:
        for t in SERIES_TYPE:
            t.url = CachedAccessor("url", UrlAccessor)
    if ImageAccessor._api_count:
        for t in SERIES_TYPE:
            t.image = CachedAccessor("image", ImageAccessor)
    if AudioAccessor._api_count:
        for t in SERIES_TYPE:
            t.audio = CachedAccessor("audio", AudioAccessor)


_install()
del _install
