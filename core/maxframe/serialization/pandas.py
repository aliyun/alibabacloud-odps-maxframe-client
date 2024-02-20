# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import datetime
import enum
from typing import Any, Dict, List

import pandas as pd
from pandas.api.extensions import ExtensionArray
from pandas.arrays import IntervalArray

from ..utils import no_default
from .core import Serializer, buffered


class DataFrameSerializer(Serializer):
    @buffered
    def serial(self, obj: pd.DataFrame, context: Dict):
        col_data = []
        for _, col in obj.items():
            if getattr(col.dtype, "hasobject", False):
                col_data.append(col.tolist())
            else:
                col_data.append(col.values)
        return [], [obj.dtypes, obj.index] + col_data, False

    def deserial(
        self, serialized: List, context: Dict, subs: List[Any]
    ) -> pd.DataFrame:
        dtypes, idx = subs[:2]
        df = pd.concat([pd.Series(d, index=idx) for d in subs[2:]], axis=1)
        df.columns = dtypes.index
        df.index = idx
        return df.astype(dtypes)


class SeriesSerializer(Serializer):
    @buffered
    def serial(self, obj: pd.Series, context: Dict):
        if getattr(obj.dtype, "hasobject", False):
            data = obj.tolist()
        else:
            data = obj.values
        return [], [obj.index, obj.name, obj.dtype, data], False

    def deserial(self, serialized: List, context: Dict, subs: List[Any]) -> pd.Series:
        idx, name, dtype, data = subs
        return pd.Series(data, index=idx, name=name).astype(dtype)


_TYPE_CHAR_MULTI_INDEX = "M"
_TYPE_CHAR_RANGE_INDEX = "R"
_TYPE_CHAR_CATEGORICAL_INDEX = "C"


class IndexSerializer(Serializer):
    @buffered
    def serial(self, obj: pd.Index, context: Dict):
        if isinstance(obj, pd.MultiIndex):
            data = [obj.get_level_values(idx) for idx in range(obj.nlevels)]
            header = [_TYPE_CHAR_MULTI_INDEX]
        elif isinstance(obj, pd.RangeIndex):
            data = [obj.name, obj.dtype]
            header = [_TYPE_CHAR_RANGE_INDEX, obj.start, obj.stop, obj.step]
        elif isinstance(obj, pd.CategoricalIndex):
            data = [obj.name, obj.values]
            header = [_TYPE_CHAR_CATEGORICAL_INDEX]
        else:
            if getattr(obj.dtype, "hasobject", False):
                values = obj.tolist()
            else:
                values = obj.values
            data = [obj.dtype, obj.name, values]
            header = [None]
        return header, data, len(data) == 0

    def deserial(self, serialized: List, context: Dict, subs: List[Any]) -> pd.Index:
        header = serialized
        if header[0] == _TYPE_CHAR_MULTI_INDEX:
            return pd.MultiIndex.from_arrays(subs)
        elif header[0] == _TYPE_CHAR_RANGE_INDEX:
            name, dtype = subs[:2]
            start, stop, step = header[1:]
            return pd.RangeIndex(start, stop, step, dtype=dtype, name=name)
        elif header[0] == _TYPE_CHAR_CATEGORICAL_INDEX:
            name, data = subs[:2]
            return pd.CategoricalIndex(data, name=name)
        elif header[0] is None:  # Normal index
            dtype, name, values = subs
            return pd.Index(values, dtype=dtype, name=name)
        else:  # pragma: no cover
            raise NotImplementedError(
                f"Deserialization for index header label {header[0]} not implemented"
            )


class CategoricalSerializer(Serializer):
    @buffered
    def serial(self, obj: pd.Categorical, context: Dict):
        return [obj.ordered], [obj.codes, obj.dtype, obj.categories], False

    def deserial(
        self, serialized: List, context: Dict, subs: List[Any]
    ) -> pd.Categorical:
        codes, dtype, categories = subs
        ordered = serialized[0]
        return pd.Categorical.from_codes(codes, categories, ordered=ordered)


_TYPE_CHAR_INTERVAL_ARRAY = "I"


class ArraySerializer(Serializer):
    @buffered
    def serial(self, obj: ExtensionArray, context: Dict):
        ser_type = None
        dtype = obj.dtype
        if isinstance(obj.dtype, pd.IntervalDtype):
            ser_type = _TYPE_CHAR_INTERVAL_ARRAY
            data_parts = [obj.left, obj.right]
        elif isinstance(obj.dtype, pd.StringDtype):
            if hasattr(obj, "tolist"):
                data_parts = [obj.tolist()]
            else:
                data_parts = [obj.to_numpy().tolist()]
        else:
            data_parts = [getattr(obj, "_data")]
        return [ser_type], [dtype] + data_parts, False

    def deserial(self, serialized: List, context: Dict, subs: List):
        if serialized[0] == _TYPE_CHAR_INTERVAL_ARRAY:
            dtype, left, right = subs
            return IntervalArray.from_arrays(left, right, dtype=dtype)
        else:
            dtype, data = subs
            return pd.array(data, dtype)


class PdTimestampSerializer(Serializer):
    def serial(self, obj: pd.Timestamp, context: Dict):
        if obj.tz:
            zone_info = [obj.tz]
            ts = obj.timestamp()
        else:
            zone_info = []
            ts = obj.to_pydatetime().timestamp()
        return (
            [int(ts), obj.microsecond, obj.nanosecond],
            zone_info,
            bool(zone_info),
        )

    def deserial(self, serialized: List, context: Dict, subs: List):
        if subs:
            val = pd.Timestamp.utcfromtimestamp(serialized[0]).replace(
                microsecond=serialized[1], nanosecond=serialized[2]
            )
            val = val.replace(tzinfo=datetime.timezone.utc).tz_convert(subs[0])
        else:
            val = pd.Timestamp.fromtimestamp(serialized[0]).replace(
                microsecond=serialized[1], nanosecond=serialized[2]
            )
        return val


class PdTimedeltaSerializer(Serializer):
    def serial(self, obj: pd.Timedelta, context: Dict):
        return [int(obj.seconds), obj.microseconds, obj.nanoseconds], [], True

    def deserial(self, serialized: List, context: Dict, subs: List):
        return pd.Timedelta(
            seconds=serialized[0], microseconds=serialized[1], nanoseconds=serialized[2]
        )


class NoDefaultSerializer(Serializer):
    def serial(self, obj: enum.Enum, context: Dict):
        return [], [], True

    def deserial(self, serialized: List, context: Dict, subs: List):
        return no_default


DataFrameSerializer.register(pd.DataFrame)
SeriesSerializer.register(pd.Series)
IndexSerializer.register(pd.Index)
CategoricalSerializer.register(pd.Categorical)
ArraySerializer.register(ExtensionArray)
PdTimestampSerializer.register(pd.Timestamp)
PdTimedeltaSerializer.register(pd.Timedelta)
NoDefaultSerializer.register(type(no_default))
