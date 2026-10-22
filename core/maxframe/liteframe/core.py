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
from maxframe.lib.dtypes_extension.blob import (
    is_polars_blob_type,
    polars_to_pandas_with_blobs,
)
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
from maxframe.tensor.datasource.from_liteframe import from_liteframe
from maxframe.utils import wrap_arrow_dtype
from maxframe.utils.polars import (
    has_pl_object_type,
    pl_get_ext_type_metadata,
    pl_get_ext_type_name_and_storage,
    pl_make_ext_or_object_dtype,
)

RANGE_COL_NAME = "__lf_range_index__"


def build_frame_from_ranges(range_columns_as_ranges):
    """Build a polars DataFrame from virtual range column metadata.

    Parameters
    ----------
    range_columns_as_ranges : dict[str, range]
        Mapping from column name to Python ``range`` object.

    Returns
    -------
    pl.DataFrame
        A DataFrame with one Int64 column per range entry. Empty if
        ``range_columns_as_ranges`` is falsy.
    """
    if not range_columns_as_ranges:
        return pl.DataFrame()
    data = {}
    for name, rng in range_columns_as_ranges.items():
        # Build from a Python list rather than ``pl.int_range`` (added in
        # polars 0.20.4) so the minimum supported polars (0.18.0) works.
        data[name] = pl.Series(name, list(rng), dtype=pl.Int64)
    return pl.DataFrame(data)


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
    """Result of a LiteFrame chunk computation.

    ``_frame`` holds the physical Polars frame (DataFrame or LazyFrame).
    When all output columns are virtual range columns (no physical data),
    ``_frame`` is ``None`` — the frame is materialized on demand from
    ``range_columns`` by :meth:`materialize`.

    Parameters
    ----------
    _frame : pl.DataFrame, pl.LazyFrame, or None
        Physical Polars frame, or ``None`` for all-virtual-range output.
    range_columns : dict[str, tuple[int, int, int]] or None
        Virtual range column metadata: name -> (start, stop, step).
    """

    _frame = AnyField("_frame")  # pl.DataFrame, pl.LazyFrame, or None
    range_columns = DictField(
        "range_columns",
        key_type=FieldTypes.string,
        value_type=FieldTypes.tuple,  # (start, stop, step)
        default=None,
    )

    @property
    def frame(self) -> "pl.DataFrame | pl.LazyFrame | None":
        """The underlying polars object, or None for all-virtual-range output."""
        return self._frame

    @property
    def range_columns_as_ranges(self) -> Dict[str, range]:
        """Convert stored (start, stop, step) tuples to Python range objects."""
        if not self.range_columns:
            return {}
        return {k: range(v[0], v[1], v[2]) for k, v in self.range_columns.items()}

    def get(self) -> "pl.DataFrame | pl.LazyFrame | None":
        """Get the polars object for computation.

        Returns ``None`` when ``_frame`` is ``None`` (all-virtual-range
        projection). Callers that need a physical frame should use
        :meth:`materialize` instead.
        """
        return self._frame

    def materialize(self) -> "pl.DataFrame | pl.LazyFrame":
        """Return a physical polars frame, materializing from range_columns if needed.

        When ``_frame`` is ``None``, builds a DataFrame from the virtual
        range columns. This is used by consumers that need a concrete
        polars object (e.g. to_odps, to_pandas, filter, sort).
        """
        if self._frame is not None:
            return self._frame
        return build_frame_from_ranges(self.range_columns_as_ranges)

    def consolidate(self) -> "pl.DataFrame | None":
        """Collect LazyFrame to DataFrame in-place and return it.

        If the frame is already a DataFrame, returns it as-is.
        If ``_frame`` is ``None``, returns ``None`` without
        materializing — use :meth:`materialize` for that.

        Note: For virtual range columns (when _frame is None), the caller
        should handle memory size estimation separately using range_columns.
        """
        if self._frame is not None and pl and isinstance(self._frame, pl.LazyFrame):
            self._frame = self._frame.collect()
        return self._frame

    def __getstate__(self):
        """Consolidate LazyFrame -> DataFrame before serialization.
        Called by Ray when sending between workers.

        Polars pl.Object columns do not survive Ray pickle — Python
        objects are replaced with raw C pointer bytes.  This covers
        pl.Object directly, pl.BaseExtension(Object), and blob
        extension types (which may have Binary storage after Arrow
        round-trip).

        To prevent corruption, all object-type column values are
        extracted into plain Python lists and stored alongside dtype
        metadata.  Pickle handles them correctly because objects like
        ODPSBlob/SolidBlob implement __reduce__.
        """
        frame = self.consolidate()

        object_data = None
        if pl and isinstance(frame, pl.DataFrame):
            for col in frame.columns:
                if not has_pl_object_type(frame[col].dtype):
                    continue
                if object_data is None:
                    object_data = {}
                ext_name, storage_str = pl_get_ext_type_name_and_storage(
                    frame[col].dtype
                )
                metadata = pl_get_ext_type_metadata(frame[col].dtype)
                object_data[col] = [
                    ext_name,
                    storage_str,
                    metadata,
                    frame[col].to_list(),
                ]
                # Replace with a null placeholder so the DataFrame
                # itself survives pickle without corruption.
                frame = frame.with_columns(
                    pl.Series(col, [None] * len(frame), dtype=pl.Binary)
                )

        state = {
            "_frame": frame,
            "range_columns": self.range_columns,
            "_object_data": object_data,
        }
        return state

    def __setstate__(self, state):
        self._frame = state["_frame"]
        self.range_columns = state.get("range_columns")

        object_data = state.get("_object_data")
        if object_data and pl and isinstance(self._frame, pl.DataFrame):
            for col, (ext_name, storage_str, metadata, values) in object_data.items():
                dtype = pl_make_ext_or_object_dtype(ext_name, storage_str, metadata)
                self._frame = self._frame.with_columns(
                    pl.Series(col, values, dtype=dtype)
                )


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
            self._physical_dtypes.drop(list(hidden), errors="ignore")
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

    @property
    def _hidden_columns(self):
        if self.frame_metadata and self.frame_metadata.hidden_columns:
            return set(self.frame_metadata.hidden_columns)
        return set()

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

    def to_tensor(self, dtype=None):
        """Convert the LiteFrame to a MaxFrame tensor.

        All columns are combined into a single homogeneous tensor. When
        ``dtype`` is not provided, the result dtype is the common
        promotion of all column dtypes.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            Target dtype for the resulting tensor. If None, the dtype is
            inferred from the column dtypes.

        Returns
        -------
        Tensor
            A MaxFrame tensor containing the LiteFrame data.

        See Also
        --------
        LiteFrame.from_tensor : Create a LiteFrame from a MaxFrame tensor.
        LiteFrame.values : Return the LiteFrame as a tensor.
        """
        return from_liteframe(self, dtype=dtype)

    def __maxframe_tensor__(self, dtype=None, order="K"):
        return self.to_tensor().astype(dtype=dtype, order=order, copy=False)

    @property
    def values(self):
        """Return the LiteFrame as a tensor.

        The LiteFrame data is converted to a homogeneous MaxFrame tensor
        whose dtype is the common promotion of all column dtypes.

        Returns
        -------
        Tensor
            A MaxFrame tensor containing the LiteFrame data.

        See Also
        --------
        LiteFrame.to_tensor : Convert the LiteFrame to a MaxFrame tensor.
        """
        return self.to_tensor()


class LiteFrameToPandasMixin(_ExecuteAndFetchMixin):
    __slots__ = ()

    def to_pandas(self, session=None, **kw):
        """Convert the LiteFrame to a pandas DataFrame.

        Executes the lazy LiteFrame and collects the results into a
        pandas DataFrame. Blob columns are preserved with their
        appropriate external blob dtypes.

        Parameters
        ----------
        session : Session, optional
            Session to use for execution. If None, the default session
            is used.
        **kw
            Additional keyword arguments forwarded to the execution
            backend.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing the materialized results.
        """
        result = self._execute_and_fetch(session=session, **kw)
        if isinstance(result, LiteFrameResult):
            # Materialize virtual range columns (when _frame is None) and
            # convert through the blob-aware bridge so blob columns come
            # back as ExternalBlobDtype / AbstractExternalBlob values.
            result = result.materialize()
        if pl and isinstance(result, pl.LazyFrame):
            result = result.collect()
        if pl and isinstance(result, pl.DataFrame):
            # Only route through the blob-aware converter when blob columns
            # are present; otherwise plain to_pandas() preserves numpy
            # dtypes (int64, not int64[pyarrow]) as the public contract.
            if any(is_polars_blob_type(result[c].dtype) for c in result.columns):
                return polars_to_pandas_with_blobs(result)
            return result.to_pandas()
        return result


class LiteFrame(LiteFrameToPandasMixin, HasShapeTileable):
    __slots__ = ("_cache",)
    _allow_data_type_ = (LiteFrameData,)
    type_name = "LiteFrame"

    def __class_getitem__(cls, item):
        from maxframe.liteframe.functional.type_infer import LiteFrameType

        return LiteFrameType.from_getitem_args(item)

    @property
    def columns(self):
        """Return the column labels of the LiteFrame.

        Returns
        -------
        list
            A list of column name strings.
        """
        return self._data.columns

    @property
    def dtypes(self):
        """Return the dtype of each column.

        Returns
        -------
        pandas.Series
            A pandas Series of dtypes indexed by column name.
        """
        return self._data.dtypes

    @property
    def ndim(self):
        """Return the number of dimensions of the LiteFrame.

        A LiteFrame is always 2-dimensional (rows and columns).

        Returns
        -------
        int
            Always returns 2.
        """
        return 2

    @property
    def frame_metadata(self):
        return self._data.frame_metadata

    @property
    def _physical_dtypes(self):
        return self._data._physical_dtypes

    @property
    def _hidden_columns(self):
        return self._data._hidden_columns

    def __hash__(self):
        return super().__hash__()

    def head(self, n=5):
        """Return the first ``n`` rows.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        LiteFrame
            A new LiteFrame containing the first ``n`` rows.

        Examples
        --------
        >>> import maxframe.liteframe as mlf
        >>> lf = mlf.LiteFrame({'a': [1, 2, 3, 4, 5]})
        >>> lf.head(3).execute()
           a
        0  1
        1  2
        2  3
        """
        from maxframe.liteframe.operators.indexing import LiteFrameIlocGetItem

        op = LiteFrameIlocGetItem(indexes=[slice(0, int(n)), slice(None)])
        return op(self)

    def to_tensor(self, dtype=None):
        """Convert the LiteFrame to a MaxFrame tensor.

        All columns are combined into a single homogeneous tensor. When
        ``dtype`` is not provided, the result dtype is the common
        promotion of all column dtypes.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            Target dtype for the resulting tensor. If None, the dtype is
            inferred from the column dtypes.

        Returns
        -------
        Tensor
            A MaxFrame tensor containing the LiteFrame data.

        See Also
        --------
        LiteFrame.from_tensor : Create a LiteFrame from a MaxFrame tensor.
        LiteFrame.values : Return the LiteFrame as a tensor.
        """
        return self._data.to_tensor(dtype=dtype)

    def __maxframe_tensor__(self, dtype=None, order="K"):
        return self._data.__maxframe_tensor__(dtype=dtype, order=order)

    @property
    def values(self):
        """Return the LiteFrame as a tensor.

        The LiteFrame data is converted to a homogeneous MaxFrame tensor
        whose dtype is the common promotion of all column dtypes.

        Returns
        -------
        Tensor
            A MaxFrame tensor containing the LiteFrame data.

        See Also
        --------
        LiteFrame.to_tensor : Convert the LiteFrame to a MaxFrame tensor.
        """
        return self.to_tensor()


LITEFRAME_TYPE = (LiteFrame, LiteFrameData)

register_output_types(OutputType.liteframe, LITEFRAME_TYPE)
