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

from typing import Dict

import pandas as pd
import pyarrow as pa

try:
    import polars as pl
except ImportError:
    pl = None

from maxframe.core import (
    HasShapeTileable,
    HasShapeTileableData,
    OutputType,
    is_build_mode,
    register_output_types,
)
from maxframe.core.entity.executable import _ExecuteAndFetchMixin
from maxframe.core.entity.utils import fill_chunk_slices, refresh_tileable_shape
from maxframe.lib.compat import cached_property
from maxframe.serialization.serializables import (
    AnyField,
    DictField,
    FieldTypes,
    Int64Field,
    ListField,
    ReferenceField,
    Serializable,
    SeriesField,
)
from maxframe.utils import wrap_arrow_dtype

RANGE_COL_NAME = "__lf_range_index__"


class RangeInfo(Serializable):
    start = Int64Field("start", default=None)
    stop = Int64Field("stop", default=None)
    step = Int64Field("step", default=None)


class FrameMetadata(Serializable):
    range_columns = DictField(
        "range_columns",
        key_type=FieldTypes.string,
        value_type=FieldTypes.reference(RangeInfo),
        default=None,
    )
    hidden_columns = ListField(
        "hidden_columns",
        FieldTypes.string,
        default=None,
    )


class LiteFrameResult(Serializable):
    _frame = AnyField("_frame")  # pl.DataFrame or pl.LazyFrame
    range_columns = DictField(
        "range_columns",
        key_type=FieldTypes.string,
        value_type=FieldTypes.tuple,  # (start, stop, step)
        default=None,
    )

    @property
    def frame(self) -> "pl.DataFrame | pl.LazyFrame":
        """The underlying polars object."""
        return self._frame

    @property
    def range_columns_as_ranges(self) -> Dict[str, range]:
        """Convert stored (start, stop, step) tuples to Python range objects."""
        if not self.range_columns:
            return {}
        return {k: range(v[0], v[1], v[2]) for k, v in self.range_columns.items()}

    def get(self) -> "pl.DataFrame | pl.LazyFrame":
        """Get the polars object for computation. Returns whatever is inside
        (DataFrame or LazyFrame) without collecting."""
        return self._frame

    def consolidate(self) -> "pl.DataFrame":
        """Collect LazyFrame to DataFrame in-place and return it.

        If the frame is already a DataFrame, returns it as-is.
        The in-place update ensures subsequent access (including
        serialization) sees the collected DataFrame.
        """
        if pl and isinstance(self._frame, pl.LazyFrame):
            self._frame = self._frame.collect()
        return self._frame

    def __getstate__(self):
        """Consolidate LazyFrame -> DataFrame before serialization.
        Called by Ray when sending between workers."""
        frame = self.consolidate()
        state = {
            "_frame": frame,
            "range_columns": self.range_columns,
        }
        return state

    def __setstate__(self, state):
        self._frame = state["_frame"]
        self.range_columns = state.get("range_columns")


class LiteFrameData(HasShapeTileableData):
    __slots__ = ("_accessors", "__dict__")
    type_name = "LiteFrame"

    _physical_dtypes = SeriesField("_physical_dtypes")
    frame_metadata = ReferenceField("frame_metadata", FrameMetadata, default=None)

    def __init__(
        self, op=None, shape=None, physical_dtypes=None, frame_metadata=None, **kw
    ):
        super().__init__(
            _op=op,
            _shape=shape,
            _physical_dtypes=physical_dtypes,
            frame_metadata=frame_metadata,
            **kw,
        )
        self._accessors = dict()

    def __on_deserialize__(self):
        super().__on_deserialize__()
        self._accessors = dict()
        self.__dict__.pop("dtypes", None)
        self.__dict__.pop("columns", None)

    @cached_property
    def columns(self):
        cols = []
        if self.frame_metadata and self.frame_metadata.range_columns:
            cols.extend(self.frame_metadata.range_columns.keys())
        hidden = (
            set(self.frame_metadata.hidden_columns)
            if (self.frame_metadata and self.frame_metadata.hidden_columns)
            else set()
        )
        cols.extend(c for c in self._physical_dtypes.index if c not in hidden)
        return cols

    @cached_property
    def dtypes(self):
        hidden = (
            set(self.frame_metadata.hidden_columns)
            if (self.frame_metadata and self.frame_metadata.hidden_columns)
            else set()
        )
        result = (
            self._physical_dtypes.drop(list(hidden))
            if hidden
            else self._physical_dtypes
        )
        if self.frame_metadata and self.frame_metadata.range_columns:
            range_dtypes = pd.Series(
                [wrap_arrow_dtype(pa.int64())] * len(self.frame_metadata.range_columns),
                index=list(self.frame_metadata.range_columns.keys()),
            )
            result = pd.concat([range_dtypes, result])
        return result

    @property
    def params(self):
        return {
            "shape": self.shape,
            "physical_dtypes": self._physical_dtypes,
            "frame_metadata": self.frame_metadata,
        }

    @params.setter
    def params(self, new_params):
        params = new_params.copy()
        new_shape = params.pop("shape", None)
        if new_shape is not None:
            self._shape = new_shape
        physical_dtypes = params.pop("physical_dtypes", None)
        if physical_dtypes is not None:
            self._physical_dtypes = physical_dtypes
            self.__dict__.pop("dtypes", None)
            self.__dict__.pop("columns", None)
        frame_metadata = params.pop("frame_metadata", None)
        if frame_metadata is not None:
            self.frame_metadata = frame_metadata
            self.__dict__.pop("dtypes", None)
            self.__dict__.pop("columns", None)
        if params:
            raise TypeError(f"Unknown params: {list(params)}")

    def refresh_params(self):
        refresh_tileable_shape(self)
        fill_chunk_slices(self)

    def _to_str(self, representation=False):
        if is_build_mode() or len(self._executed_sessions) == 0:
            if representation:
                return (
                    f"{self.type_name} <op={type(self._op).__name__}, key={self.key}>"
                )
            else:
                return f"{self.type_name}(op={type(self._op).__name__})"
        else:
            data = self._fetch(session=self._executed_sessions[-1])
            if isinstance(data, LiteFrameResult):
                data = data.frame
            if pl and isinstance(data, pl.LazyFrame):
                data = data.collect()
            return repr(data) if representation else str(data)

    def __str__(self):
        return self._to_str(representation=False)

    def __repr__(self):
        return self._to_str(representation=True)


class LiteFrameToPandasMixin(_ExecuteAndFetchMixin):
    __slots__ = ()

    def to_pandas(self, session=None, **kw):
        result = self._execute_and_fetch(session=session, **kw)
        if isinstance(result, LiteFrameResult):
            result = result.frame
        if pl and isinstance(result, pl.LazyFrame):
            result = result.collect()
        if pl and isinstance(result, pl.DataFrame):
            return result.to_pandas()
        return result


class LiteFrame(LiteFrameToPandasMixin, HasShapeTileable):
    __slots__ = ("_cache",)
    _allow_data_type_ = (LiteFrameData,)
    type_name = "LiteFrame"

    @property
    def columns(self):
        return self._data.columns

    @property
    def dtypes(self):
        return self._data.dtypes

    @property
    def ndim(self):
        return 2

    @property
    def frame_metadata(self):
        return self._data.frame_metadata

    @property
    def _physical_dtypes(self):
        return self._data._physical_dtypes

    @property
    def _hidden_columns(self):
        if self._data.frame_metadata and self._data.frame_metadata.hidden_columns:
            return set(self._data.frame_metadata.hidden_columns)
        return set()

    def __hash__(self):
        return super().__hash__()

    def head(self, n=5):
        from maxframe.liteframe.indexing.iloc import LiteFrameIlocGetItem

        op = LiteFrameIlocGetItem(indexes=[slice(0, int(n)), slice(None)])
        return op(self)


LITEFRAME_TYPE = (LiteFrame, LiteFrameData)

register_output_types(OutputType.liteframe, LITEFRAME_TYPE)
