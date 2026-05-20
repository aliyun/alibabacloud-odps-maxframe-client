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

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from maxframe.lib.dtypes_extension import list_
from maxframe.serialization.serializables import (
    BytesField,
    Float64Field,
    Int32Field,
    Serializable,
    StringField,
)


class AudioObject(Serializable):
    data = BytesField("data", default=None)
    sample_rate = Int32Field("sample_rate", default=None)
    duration = Float64Field("duration", default=None)
    channels = Int32Field("channels", default=None)
    format = StringField("format", default=None)

    @property
    def size(self) -> int:
        return len(self.data) if self.data else 0


@dataclass
class TranscribeResult:
    text: Optional[str] = None
    language: Optional[str] = None
    language_probability: Optional[float] = None
    segments: Optional[str] = None

    @classmethod
    def output_dtypes(cls) -> pd.Series:
        return pd.Series(
            {
                "text": np.dtype("object"),
                "language": np.dtype("object"),
                "language_probability": np.dtype("float64"),
                "segments": np.dtype("object"),
            }
        )

    def to_series(self) -> pd.Series:
        return pd.Series(
            {
                "text": self.text,
                "language": self.language,
                "language_probability": (
                    np.nan
                    if self.language_probability is None
                    else self.language_probability
                ),
                "segments": self.segments,
            },
            dtype=object,
        )


@dataclass
class VadSegment:
    start: float = 0.0
    end: float = 0.0
    confidence: Optional[float] = None

    @staticmethod
    @lru_cache(maxsize=None)
    def dtype():
        return list_(
            pa.struct(
                [
                    pa.field("start", pa.float64()),
                    pa.field("end", pa.float64()),
                    pa.field("confidence", pa.float64()),
                ]
            )
        )
