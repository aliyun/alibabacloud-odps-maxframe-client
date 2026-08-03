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

import pandas as pd

from maxframe.liteframe.core import LiteFrame as _LiteFrame
from maxframe.serialization.serializables import SerializableMeta


class InitializerMeta(SerializableMeta):
    def __instancecheck__(cls, instance):
        return isinstance(
            instance, (cls.__base__,) + getattr(cls, "_allow_data_type_", ())
        )


class LiteFrame(_LiteFrame, metaclass=InitializerMeta):
    def __init__(
        self,
        data=None,
        columns=None,
        dtype=None,
        chunk_size=None,
        gpu=None,
    ):
        # Lazy import to avoid circular dependency with datasource module
        from maxframe.liteframe.datasource.from_local import from_local_df

        if isinstance(data, _LiteFrame):
            super().__init__(data.data)
            return

        if isinstance(data, dict):
            pdf = pd.DataFrame(data, columns=columns, dtype=dtype)
        elif isinstance(data, pd.DataFrame):
            pdf = data.copy()
            if columns is not None:
                pdf = pdf[columns]
        elif data is None:
            pdf = pd.DataFrame(columns=columns or [], dtype=dtype)
        else:
            pdf = pd.DataFrame(data, columns=columns, dtype=dtype)

        lf = from_local_df(pdf, chunk_size=chunk_size, gpu=gpu)
        super().__init__(lf.data)
