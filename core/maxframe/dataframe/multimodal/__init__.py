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
    from maxframe.dataframe.core import register_series_accessor

    register_series_accessor("url")(UrlAccessor)
    register_series_accessor("image")(ImageAccessor)
    register_series_accessor("audio")(AudioAccessor)


_install()
del _install
