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

import logging
from typing import List, MutableMapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from odps import ODPS
from odps.errors import NoSuchObject
from odps.models import Table
from odps.utils import to_timestamp

from maxframe import opcodes
from maxframe.config import option_context, options
from maxframe.core import OutputType
from maxframe.dataframe.core import DataFrame  # noqa: F401
from maxframe.dataframe.datasource.core import (
    ColumnPruneSupportedDataSourceMixin,
    DtypeBackendCompatibleMixin,
    IncrementalIndexDatasource,
)
from maxframe.dataframe.utils import (
    get_index_value_by_default_index_type,
    parse_index,
    validate_default_index_type,
    validate_dtype_backend,
)
from maxframe.io.odpsio import odps_schema_to_pandas_dtypes
from maxframe.protocol import DefaultIndexType
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    EnumField,
    FieldTypes,
    Int64Field,
    ListField,
    SeriesField,
    StringField,
)
from maxframe.utils import (
    deprecate_positional_args,
    estimate_table_size,
    get_odps_dlf_table,
    is_empty,
    str_to_bool,
)

logger = logging.getLogger(__name__)


class DataFrameReadODPSTable(
    IncrementalIndexDatasource,
    ColumnPruneSupportedDataSourceMixin,
    DtypeBackendCompatibleMixin,
):
    __slots__ = ("_odps_entry",)
    _op_type_ = opcodes.READ_ODPS_TABLE

    table_name = StringField("table_name")
    partitions = ListField("partitions", field_type=FieldTypes.string, default=None)
    dtypes = SeriesField("dtypes", default=None)
    columns = AnyField("columns", default=None)
    nrows = Int64Field("nrows", default=None)
    dtype_backend = StringField("dtype_backend", default=None)
    string_as_binary = BoolField("string_as_binary", default=None)
    append_partitions = BoolField("append_partitions", default=None)
    last_modified_time = Int64Field("last_modified_time", default=None)
    index_columns = ListField("index_columns", FieldTypes.string, default=None)
    index_dtypes = SeriesField("index_dtypes", default=None)
    default_index_type = EnumField(
        "default_index_type", DefaultIndexType, FieldTypes.int8, default=None
    )
    filters = AnyField("filters", default=None)

    def __init__(self, memory_scale=None, dtype_backend=None, **kw):
        output_type = kw.pop("output_type", OutputType.dataframe)
        self._odps_entry = kw.pop("odps_entry", None)
        dtype_backend = validate_dtype_backend(dtype_backend)
        super(DataFrameReadODPSTable, self).__init__(
            memory_scale=memory_scale,
            dtype_backend=dtype_backend,
            _output_types=[output_type],
            **kw,
        )

    @property
    def partition(self):
        return getattr(self, "partition_spec", None)

    def can_fuse_with_custom_code(self) -> bool:
        return False

    def get_columns(self):
        return self.columns or list(self.dtypes.index)

    def set_pruned_columns(self, columns, *, keep_order=None):  # pragma: no cover
        self.columns = columns

    def __call__(self, shape, chunk_bytes=None, chunk_size=None):
        if is_empty(self.index_columns):
            index_value = get_index_value_by_default_index_type(
                self.default_index_type, shape[0], args=(type(self), self.table_name)
            )
        elif len(self.index_columns) == 1:
            index_value = parse_index(
                pd.Index([], name=self.index_columns[0]).astype(
                    self.index_dtypes.iloc[0]
                )
            )
        else:
            idx = pd.MultiIndex.from_frame(
                pd.DataFrame([], columns=self.index_columns).astype(self.index_dtypes)
            )
            index_value = parse_index(idx)

        if self.output_types[0] == OutputType.dataframe:
            columns_value = parse_index(self.dtypes.index, store_data=True)
            return self.new_tileable(
                [],
                None,
                shape=shape,
                dtypes=self.dtypes,
                index_value=index_value,
                columns_value=columns_value,
                chunk_bytes=chunk_bytes,
                chunk_size=chunk_size,
            )
        elif self.output_types[0] == OutputType.series:
            return self.new_tileable(
                [],
                None,
                shape=shape[:1],
                name=self.dtypes.index[0],
                dtype=self.dtypes.iloc[0],
                index_value=index_value,
                chunk_bytes=chunk_bytes,
                chunk_size=chunk_size,
            )
        elif self.output_types[0] == OutputType.index:
            return self.new_tileable(
                [],
                None,
                shape=shape[:1],
                name=getattr(index_value, "name", None),
                names=getattr(index_value, "names", None),
                dtype=self.index_dtypes.iloc[0],
                index_value=index_value,
                chunk_bytes=chunk_bytes,
                chunk_size=chunk_size,
            )

    @classmethod
    def estimate_size(
        cls, ctx: MutableMapping[str, Union[int, float]], op: "DataFrameReadODPSTable"
    ) -> None:
        odps_entry = op._odps_entry or ODPS.from_global() or ODPS.from_environments()
        if not odps_entry:  # pragma: no cover
            ctx[op.outputs[0].key] = float("inf")
            return
        ctx[op.outputs[0].key] = estimate_table_size(
            odps_entry, op.table_name, op.partitions
        )


@deprecate_positional_args
def read_odps_table(
    table_name: Union[str, Table],
    partitions: Union[None, str, List[str]] = None,
    columns: Optional[List[str]] = None,
    index_col: Union[None, str, List[str]] = None,
    *,
    odps_entry: ODPS = None,
    string_as_binary: bool = None,
    append_partitions: bool = False,
    dtype_backend: str = None,
    default_index_type: DefaultIndexType = None,
    filters: Union[str, List[List[Tuple]]] = None,
    **kw,
):
    """
    Read a MaxCompute (ODPS) table into a DataFrame.

    Use ``index_col`` to select index columns. Otherwise the engine generates
    an index using the configured default strategy.

    Parameters
    ----------
    table_name : str or odps.models.Table
        Name or object of the table to read.
    partitions : str or list of str, optional
        Partition specification or list of specifications to read. If omitted,
        read all partitions.
    columns : list of str, optional
        Columns to read, including partition columns when needed. If omitted,
        read all data columns except those selected as indexes.
    index_col : str or list of str, optional
        Columns to use as the index. These cannot also appear in ``columns``.
    odps_entry : odps.ODPS, optional
        Connection used to resolve and read the table. If omitted, use the
        global ODPS connection, then the connection from environment variables.
    string_as_binary : bool, optional
        Whether to read string values as binary. If None, use the reader's
        default string handling.
    append_partitions : bool, default False
        Include partition columns when ``columns`` is omitted. Selecting a
        partition column explicitly also enables partition-column reading.
    dtype_backend : {'numpy', 'pyarrow'}, optional
        Result dtype backend. Defaults to ``options.dataframe.dtype_backend``
        when None. This option is experimental.
    default_index_type : {'range', 'incremental', 'chunk_range'}, optional
        Index strategy when ``index_col`` is omitted. Defaults to
        ``options.dataframe.default_index_type`` (``'range'`` by default).
        With DPE, ``'range'`` produces consecutive integers starting at zero;
        ``'incremental'`` produces increasing integers with gaps between chunks;
        ``'chunk_range'`` restarts at zero in each chunk and can produce duplicate
        index values. These chunk-based strategies do not control MCSQL index
        generation.
    filters : str or list of list of tuple, optional
        Filter passed to the reader. A string is a SQL WHERE expression.
        In the nested-list form, conditions in each inner list are ANDed;
        the outer lists are ORed. For example,
        ``[[('col1', '==', 'value'), ('col2', '>', 10)]]``.
        Supported operators are ``==``, ``!=``, ``<``, ``>``, ``<=``, ``>=``,
        ``in``, and ``not in``. Complete filtering is not guaranteed.
    **kw
        Additional read options. Common options are listed below.

    Other Parameters
    ----------------
    chunk_bytes : int, optional
        Target source bytes per input split, passed through ``**kw``. DPE uses
        it for non-transactional tables when ``nrows`` is omitted; the default
        is 64 MiB. MCSQL converts an explicit value to MiB with
        ``max(1, chunk_bytes // 1024**2)`` and sets the SQL ``split_size`` hint.
        SPE ignores this setting.
    chunk_size : int, optional
        Target rows per DPE input chunk for transactional tables or reads with
        ``nrows``. Passed through ``**kw``. Defaults to ``options.chunk_size``
        when configured, otherwise 1024 rows. MCSQL and SPE ignore this setting.
    nrows : int, optional
        Maximum rows to read, passed through ``**kw``. If omitted, no row limit
        is requested.

    Returns
    -------
    DataFrame
        Lazy DataFrame representing the selected table data.

    Raises
    ------
    ValueError
        If no ODPS connection is available or a requested column is missing.
        Also raised for partition options on a non-partitioned table,
        overlapping index and data columns, or an invalid index strategy or
        dtype backend.
    TypeError
        If PyODPS cannot construct a connection from incomplete environment
        credentials. Pass an initialized ODPS object with ``odps_entry`` to
        avoid environment-based connection discovery.
    odps.errors.NoSuchObject
        If the table cannot be found or resolved through the configured catalog.

    See Also
    --------
    DataFrame.to_odps_table : Write a DataFrame to a MaxCompute table.

    Notes
    -----
    For input partition sizes, see :ref:`udf_source_chunk_settings`.
    For task memory and CPU sizing, see :ref:`udf_tuning_resources`.

    Examples
    --------
    These examples require a MaxCompute connection and existing tables with
    the columns and partitions shown. Set the connection environment variables
    before running them. Reading creates a lazy expression; execution is separate.

    >>> import os
    >>> import maxframe.dataframe as md
    >>> from maxframe.session import new_session
    >>> from odps import ODPS
    >>> o = ODPS(  # doctest: +SKIP
    ...     os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
    ...     os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
    ...     project=os.environ["ODPS_PROJECT"],
    ...     endpoint=os.environ["ODPS_ENDPOINT"],
    ... )

    Read a table with a target source split size of 32 MiB.

    >>> df = md.read_odps_table(  # doctest: +SKIP
    ...     "simple_table", odps_entry=o, chunk_bytes=32 * 1024 * 1024
    ... )
    >>> session = new_session(o)  # doctest: +SKIP
    >>> try:  # doctest: +SKIP
    ...     result = df.execute(session=session).fetch()
    ... finally:
    ...     session.destroy()

    Limit the read to 10,000 rows and request 1,000-row DPE input chunks.

    >>> sample = md.read_odps_table(  # doctest: +SKIP
    ...     "simple_table", odps_entry=o, nrows=10_000, chunk_size=1000
    ... )

    Select partitions, data columns, and an index column.

    >>> df = md.read_odps_table(  # doctest: +SKIP
    ...     "partitioned_table", odps_entry=o,
    ...     partitions=["pt=20230101", "pt=20230102"],
    ...     columns=["name", "age"], index_col="id",
    ... )

    Pass a SQL filter expression, or the equivalent nested-list form.
    Complete filtering is not guaranteed.

    >>> df = md.read_odps_table(  # doctest: +SKIP
    ...     "source_table", odps_entry=o, filters="age > 18"
    ... )
    >>> df = md.read_odps_table(  # doctest: +SKIP
    ...     "source_table", odps_entry=o, filters=[[("age", ">", 18)]]
    ... )
    """
    default_index_type = validate_default_index_type(default_index_type, **kw)
    odps_entry = odps_entry or ODPS.from_global() or ODPS.from_environments()
    if odps_entry is None:
        raise ValueError("Missing odps_entry parameter")
    schema = options.session.default_schema or odps_entry.schema
    if isinstance(table_name, Table):
        table = table_name
    else:
        try:
            table = odps_entry.get_table(table_name, schema=schema)
            table.reload()
        except NoSuchObject:
            sql_hints = options.sql.settings or {}
            if not str_to_bool(sql_hints.get("odps.maxframe.resolve_dlf_tables")):
                raise
            if isinstance(table_name, Table):
                table_name = table_name.name
            table = get_odps_dlf_table(odps_entry, table_name, schema=schema)

    if not table.table_schema.partitions and (
        partitions is not None or append_partitions
    ):
        raise ValueError("Cannot handle partitions on non-partition tables")

    cols = (
        table.table_schema.columns
        if append_partitions or columns is not None
        else table.table_schema.simple_columns
    )
    table_columns = [c.name.lower() for c in cols]

    if "use_arrow_dtype" in kw:
        dtype_backend = dtype_backend or validate_dtype_backend(
            kw.pop("use_arrow_dtype")
        )
    dtype_backend = validate_dtype_backend(
        dtype_backend or options.dataframe.dtype_backend
    )

    with option_context():
        options.dataframe.dtype_backend = dtype_backend
        table_dtypes = odps_schema_to_pandas_dtypes(
            table.table_schema, with_partitions=True
        )
    df_types = [table_dtypes[c] for c in table_columns]

    if isinstance(index_col, str):
        index_col = [index_col]
    if index_col and columns is None:
        index_col_set = set([c.lower() for c in index_col])
        col_diff = sorted(index_col_set - set(table_columns))
        if col_diff:
            raise ValueError(
                f"Cannot found column {', '.join(col_diff)} specified "
                f"in index_columns argument in source table"
            )
        columns = [c for c in table_columns if c not in index_col_set]

    if not index_col:
        index_dtypes = None
    else:
        table_index_types = [df_types[table_columns.index(col)] for col in index_col]
        index_dtypes = pd.Series(table_index_types, index=index_col)

    if columns is not None:
        new_columns = [c.lower() for c in columns]
        table_col_set = set(new_columns)
        col_diff = sorted(table_col_set - set(table_columns))
        if col_diff:
            raise ValueError(
                f"Cannot found column {', '.join(col_diff)} specified "
                f"in columns argument in source table"
            )
        if any(col in table_col_set for col in index_col or ()):
            raise ValueError("Index columns and columns shall not overlap.")

        # reorder columns
        df_types = [df_types[table_columns.index(col)] for col in new_columns]
        table_columns = new_columns
        columns = new_columns

    shape = kw.pop("shape", None) or (np.nan, len(df_types))
    dtypes = pd.Series(df_types, index=table_columns)
    chunk_bytes = kw.pop("chunk_bytes", None)
    chunk_size = kw.pop("chunk_size", None)

    partitions = partitions or kw.get("partition")
    if isinstance(partitions, str):
        partitions = [partitions]

    append_partitions = append_partitions or any(
        pt.name in (columns if not is_empty(columns) else ())
        for pt in (table.table_schema.partitions or ())
    )
    op = DataFrameReadODPSTable(
        table_name=table.full_table_name,
        partitions=partitions,
        dtypes=dtypes,
        columns=columns,
        dtype_backend=dtype_backend,
        string_as_binary=string_as_binary,
        append_partitions=append_partitions,
        last_modified_time=to_timestamp(table.last_data_modified_time),
        index_columns=index_col,
        index_dtypes=index_dtypes,
        odps_entry=odps_entry,
        default_index_type=default_index_type,
        filters=filters,
        **kw,
    )
    return op(shape, chunk_bytes=chunk_bytes, chunk_size=chunk_size)
