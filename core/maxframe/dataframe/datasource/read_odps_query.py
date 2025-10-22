# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

import dataclasses
import functools
import io
import logging
import re
import tokenize
from typing import Dict, List, MutableMapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from odps import ODPS
from odps.errors import ODPSError
from odps.models import TableSchema
from odps.types import Column, OdpsSchema, validate_data_type
from odps.utils import split_sql_by_semicolon

from ... import opcodes
from ...config import option_context, options
from ...core import OutputType
from ...core.graph import DAG
from ...io.odpsio import odps_schema_to_pandas_dtypes
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    FieldTypes,
    Int64Field,
    ListField,
    SeriesField,
    StringField,
)
from ...utils import is_empty
from ..utils import parse_index, validate_dtype_backend
from .core import (
    ColumnPruneSupportedDataSourceMixin,
    DtypeBackendCompatibleMixin,
    IncrementalIndexDatasource,
)

logger = logging.getLogger(__name__)

_DEFAULT_ANONYMOUS_COL_PREFIX = "_anon_col_"

_EXPLAIN_DEPENDS_REGEX = re.compile(r"([^\s]+) depends on: ([^\n]+)")
_EXPLAIN_JOB_REGEX = re.compile(r"(\S+) is root job")
_EXPLAIN_TASKS_HEADER_REGEX = re.compile(r"In Job ([^:]+):")
_EXPLAIN_ROOT_TASKS_REGEX = re.compile(r"root Tasks: (.+)")
_EXPLAIN_TASK_REGEX = re.compile(r"In Task ([^:]+)")
_EXPLAIN_TASK_SCHEMA_REGEX = re.compile(
    r"In Task ([^:]+)[\S\s]+FS: output: ([^\n #]+)[\s\S]+schema:\s+([\S\s]+)$",
    re.MULTILINE,
)
_EXPLAIN_COLUMN_REGEX = re.compile(r"([^\(]+) \(([^\n]+)\)(?:| AS ([^ ]+))(?:\n|$)")
_ANONYMOUS_COL_REGEX = re.compile(r"^_c(\d+)$")

_SIMPLE_SCHEMA_COLS_REGEX = re.compile(r"SELECT (([^:]+:[^, ]+[, ]*)+)FROM")
_SIMPLE_SCHEMA_COL_REGEX = re.compile(r"([^ \.\)]+):([^ ]+)")


@dataclasses.dataclass
class DependencySector:
    roots: List[str]
    dependencies: List[Tuple[str, str]]

    def build_dag(self) -> DAG:
        dag = DAG()
        for r in self.roots:
            dag.add_node(r)
        for v_from, v_to in self.dependencies:
            dag.add_node(v_from)
            dag.add_node(v_to)
            dag.add_edge(v_from, v_to)
        return dag


@dataclasses.dataclass
class JobsSector(DependencySector):
    jobs: Dict[str, "TasksSector"] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TasksSector(DependencySector):
    job_name: str
    tasks: Dict[str, "TaskSector"] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ColumnSchema:
    column_name: str
    column_type: str
    column_alias: Optional[str]


@dataclasses.dataclass
class TaskSector:
    job_name: str
    task_name: str
    output_target: Optional[str]
    schema: List[ColumnSchema]


def _split_explain_string(explain_string: str) -> List[str]:
    parts = explain_string.split("\n\n")
    final_parts = []
    grouped = []
    for part in parts:
        part = part.strip("\n")
        part_line1 = part.split("\n", 1)[0]
        # initial line of part should not start with spaces (Statistics row)
        #  or with quote marks
        if (
            grouped
            and not part.startswith(" ")
            and "'" not in part_line1
            and '"' not in part_line1
        ):
            final_parts.append("\n\n".join(grouped).strip())
            grouped = []
        grouped.append(part)
    if grouped:
        final_parts.append("\n\n".join(grouped).strip())
    return final_parts


def _find_all_deps(sector: str) -> List[Tuple[str, str]]:
    deps = []
    for match in _EXPLAIN_DEPENDS_REGEX.findall(sector):
        descendant = match[0]
        for r in match[1].split(","):
            deps.append((r.strip(), descendant))
    return deps


def _resolve_jobs_sector(sector: str) -> JobsSector:
    match = _EXPLAIN_JOB_REGEX.search(sector)
    roots = [r.strip() for r in match.group(1).split(",")]
    deps = _find_all_deps(sector)
    return JobsSector(roots, deps)


def _resolve_tasks_sector(sector: str) -> TasksSector:
    match = _EXPLAIN_ROOT_TASKS_REGEX.search(sector)
    roots = [r.strip() for r in match.group(1).split(",")]

    match = _EXPLAIN_TASKS_HEADER_REGEX.search(sector)
    job_name = match.group(1)

    deps = _find_all_deps(sector)
    return TasksSector(roots, deps, job_name)


def _resolve_task_sector(job_name: str, sector: str) -> TaskSector:
    match = _EXPLAIN_TASK_REGEX.match(sector)
    task_name = match.group(1)

    match = _EXPLAIN_TASK_SCHEMA_REGEX.match(sector)
    if match is None:
        return TaskSector(job_name, task_name, None, [])

    out_target = match.group(2)
    out_schema = match.group(3)

    schemas = []
    for match in _EXPLAIN_COLUMN_REGEX.findall(out_schema):
        col_name, data_type, alias = match
        schemas.append(ColumnSchema(col_name.strip(), data_type.strip(), alias.strip()))
    return TaskSector(job_name, task_name, out_target, schemas)


def _select_task_prefix(sector: TasksSector, prefix: str) -> List[TaskSector]:
    if prefix in sector.tasks:
        return [sector.tasks[prefix]]
    return [v for k, v in sector.tasks.items() if k.startswith(prefix + "_")]


def _parse_full_explain(explain_string: str) -> OdpsSchema:
    sectors = _split_explain_string(explain_string)
    jobs_sector = tasks_sector = None

    for sector in sectors:
        if _EXPLAIN_JOB_REGEX.search(sector):
            jobs_sector = _resolve_jobs_sector(sector)
        elif _EXPLAIN_TASKS_HEADER_REGEX.search(sector):
            tasks_sector = _resolve_tasks_sector(sector)
            assert jobs_sector is not None
            jobs_sector.jobs[tasks_sector.job_name] = tasks_sector
        elif _EXPLAIN_TASK_REGEX.search(sector):
            assert tasks_sector is not None
            task_sector = _resolve_task_sector(tasks_sector.job_name, sector)
            tasks_sector.tasks[task_sector.task_name] = task_sector

    job_dag = jobs_sector.build_dag()
    indep_job_names = list(job_dag.iter_indep(reverse=True))
    schema_signatures = dict()
    for job_name in indep_job_names:
        tasks_sector = jobs_sector.jobs[job_name]
        task_dag = tasks_sector.build_dag()
        indep_task_names = list(task_dag.iter_indep(reverse=True))
        for task_name in indep_task_names:
            for task_sector in _select_task_prefix(tasks_sector, task_name):
                if not task_sector.schema:  # pragma: no cover
                    raise ValueError("Cannot detect output schema")
                if task_sector.output_target != "Screen":
                    raise ValueError("The SQL statement should be an instant query")
                sig_tuples = sorted(
                    [
                        (c.column_alias or c.column_name, c.column_type)
                        for c in task_sector.schema
                    ]
                )
                schema_signatures[hash(tuple(sig_tuples))] = task_sector.schema
    if len(schema_signatures) != 1:
        raise ValueError("Only one final task is allowed in SQL statement")
    schema = list(schema_signatures.values())[0]
    cols = [
        Column(c.column_alias or c.column_name, validate_data_type(c.column_type))
        for c in schema
    ]
    return OdpsSchema(cols)


def _parse_simple_explain(explain_string: str) -> OdpsSchema:
    fields_match = _SIMPLE_SCHEMA_COLS_REGEX.search(explain_string)
    if not fields_match:
        raise ValueError("Cannot detect output table schema")

    fields_str = fields_match.group(1)
    cols = []
    for field, type_name in _SIMPLE_SCHEMA_COL_REGEX.findall(fields_str):
        cols.append(Column(field, validate_data_type(type_name.rstrip(","))))
    return OdpsSchema(cols)


def _parse_explained_schema(explain_string: str) -> OdpsSchema:
    if explain_string.startswith("AdhocSink"):
        return _parse_simple_explain(explain_string)
    else:
        return _parse_full_explain(explain_string)


def _build_explain_sql(
    sql_stmt: str, no_split: bool = False, use_output: bool = False
) -> str:
    clause = "EXPLAIN "
    if use_output:
        clause += "OUTPUT "
    if no_split:
        return clause + sql_stmt
    sql_parts = split_sql_by_semicolon(sql_stmt)
    if not sql_parts:
        raise ValueError(f"Cannot explain SQL statement {sql_stmt}")
    sql_parts[-1] = clause + sql_parts[-1]
    return "\n".join(sql_parts)


class DataFrameReadODPSQuery(
    IncrementalIndexDatasource,
    ColumnPruneSupportedDataSourceMixin,
    DtypeBackendCompatibleMixin,
):
    _op_type_ = opcodes.READ_ODPS_QUERY

    query = StringField("query")
    dtypes = SeriesField("dtypes", default=None)
    columns = AnyField("columns", default=None)
    nrows = Int64Field("nrows", default=None)
    dtype_backend = StringField("dtype_backend", default=None)
    string_as_binary = BoolField("string_as_binary", default=None)
    index_columns = ListField("index_columns", FieldTypes.string, default=None)
    index_dtypes = SeriesField("index_dtypes", default=None)
    column_renames = DictField("column_renames", default=None)

    def __init__(self, dtype_backend=None, **kw):
        dtype_backend = validate_dtype_backend(dtype_backend)
        super().__init__(dtype_backend=dtype_backend, **kw)

    def get_columns(self):
        return self.columns or list(self.dtypes.index)

    def set_pruned_columns(self, columns, *, keep_order=None):  # pragma: no cover
        self.columns = columns

    @classmethod
    def estimate_size(
        cls, ctx: MutableMapping[str, Union[int, float]], op: "DataFrameReadODPSQuery"
    ):  # pragma: no cover
        # use infinity to show that the size cannot be inferred
        # todo when local catalyst is implemented,
        #  a more precise estimation here can be useful then.
        ctx[op.outputs[0].key] = float("inf")

    def __call__(self, chunk_bytes=None, chunk_size=None):
        if is_empty(self.index_columns):
            index_value = parse_index(pd.RangeIndex(0))
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

        if self.dtypes is not None:
            columns_value = parse_index(self.dtypes.index, store_data=True)
            shape = (np.nan, len(self.dtypes))
        else:
            columns_value = None
            shape = (np.nan, np.nan)

        self.output_types = [OutputType.dataframe]
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


def _check_token_in_sql(token: str, sql: str) -> bool:
    try:
        names = set()
        for tk_info in tokenize.tokenize(io.BytesIO(sql.encode()).readline):
            if tk_info.type == tokenize.NAME:
                names.add(tk_info.string)
        return token in names
    except:  # pragma: no cover
        return False


def _resolve_schema_by_explain(
    odps_entry: ODPS,
    query: str,
    no_split_sql: bool = False,
    hints: Dict[str, str] = None,
    use_explain_output: bool = True,
) -> OdpsSchema:
    hints = (hints or dict()).copy()
    hints["odps.sql.select.output.format"] = "json"
    explain_stmt = _build_explain_sql(
        query, no_split=no_split_sql, use_output=use_explain_output
    )
    inst = odps_entry.execute_sql(explain_stmt, hints=hints)
    logger.debug("Explain output instance ID: %s", inst.id)
    explain_str = list(inst.get_task_results().values())[0]
    if use_explain_output:
        if not explain_str or "nothing to explain" in explain_str:
            raise ValueError("The SQL statement should be an instant query")
        return TableSchema.parse(None, explain_str)
    else:
        return _parse_explained_schema(explain_str)


def _resolve_query_schema(
    odps_entry: ODPS,
    query: str,
    no_split_sql: bool = False,
    hints: Dict[str, str] = None,
    use_explain_output: Optional[bool] = None,
) -> OdpsSchema:
    methods = []
    if use_explain_output is not False:
        # None or True
        methods.append(_resolve_schema_by_explain)
    if use_explain_output is not True:
        # None or False
        methods.append(
            functools.partial(_resolve_schema_by_explain, use_explain_output=False)
        )
    for idx, resolve_method in enumerate(methods):
        try:
            return resolve_method(
                odps_entry, query, no_split_sql=no_split_sql, hints=hints
            )
        except ODPSError as ex:
            msg = (
                f"Failed to obtain schema from SQL explain: {ex!r}\n"
                f"Explain instance ID: {ex.instance_id}"
            )
            if idx + 1 == len(methods) or "ODPS-0130161" not in str(ex):
                exc = ValueError(msg)
                raise exc.with_traceback(ex.__traceback__) from None
    # will this happen?
    raise ValueError("Failed to obtain schema from SQL explain")  # pragma: no cover


def read_odps_query(
    query: str,
    odps_entry: ODPS = None,
    index_col: Union[None, str, List[str]] = None,
    string_as_binary: bool = None,
    sql_hints: Dict[str, str] = None,
    anonymous_col_prefix: str = _DEFAULT_ANONYMOUS_COL_PREFIX,
    skip_schema: bool = False,
    dtype_backend: str = None,
    **kw,
):
    """
    Read data from a MaxCompute (ODPS) query into DataFrame.

    Supports specifying some columns as indexes. If not specified, RangeIndex
    will be generated.

    Parameters
    ----------
    query: str
        MaxCompute SQL statement.
    index_col: Union[None, str, List[str]]
        Columns to be specified as indexes.
    string_as_binary: bool, optional
        Whether to convert string columns to binary.
    sql_hints: Dict[str, str], optional
        User specified SQL hints.
    anonymous_col_prefix: str, optional
        Prefix for anonymous columns, '_anon_col_' by default.
    skip_schema: bool, optional
        Skip resolving output schema before execution. Once this is configured,
        the output DataFrame cannot be inputs of other DataFrame operators
        before execution.
    dtype_backend: {'numpy', 'pyarrow'}, default 'numpy'
        Back-end data type applied to the resultant DataFrame (still experimental).

    Returns
    -------
    result: DataFrame
        DataFrame read from MaxCompute (ODPS) table
    """
    no_split_sql = kw.pop("no_split_sql", False)
    # if use_explain_output is None, will try two methods.
    use_explain_output = kw.pop("use_explain_output", None)

    hints = options.sql.settings.copy() or {}
    if sql_hints:
        hints.update(sql_hints)

    odps_entry = odps_entry or ODPS.from_global() or ODPS.from_environments()
    if odps_entry is None:
        raise ValueError(
            "Need to provide an odps_entry argument or hold a default ODPS entry."
        )

    if options.session.enable_schema or odps_entry.is_schema_namespace_enabled():
        hints["odps.namespace.schema"] = "true"
        hints["odps.sql.allow.namespace.schema"] = "true"

    hints["odps.sql.submit.mode"] = "script"
    # fixme workaround for multi-stage split process
    hints["odps.sql.object.table.split.by.object.size.enabled"] = "false"

    if odps_entry is None:
        raise ValueError("Missing odps_entry parameter")

    if "use_arrow_dtype" in kw:
        dtype_backend = dtype_backend or validate_dtype_backend(
            kw.pop("use_arrow_dtype")
        )
    dtype_backend = validate_dtype_backend(
        dtype_backend or options.dataframe.dtype_backend
    )

    col_renames = {}
    if not skip_schema:
        odps_schema = _resolve_query_schema(
            odps_entry,
            query,
            no_split_sql=no_split_sql,
            hints=hints,
            use_explain_output=use_explain_output,
        )

        new_columns = []
        for col in odps_schema.columns:
            anon_match = _ANONYMOUS_COL_REGEX.match(col.name)
            if anon_match and not _check_token_in_sql(col.name, query):
                new_name = anonymous_col_prefix + anon_match.group(1)
                col_renames[col.name] = new_name
                new_columns.append(Column(new_name, col.type))
            else:
                new_columns.append(col)

        with option_context():
            options.dataframe.dtype_backend = dtype_backend
            dtypes = odps_schema_to_pandas_dtypes(OdpsSchema(new_columns))
    else:
        dtypes = None

    if not index_col:
        index_dtypes = None
    else:
        if dtypes is None:
            raise ValueError("Cannot configure index_col when skip_schema is True")

        if isinstance(index_col, str):
            index_col = [index_col]
        index_col_set = set(index_col)
        data_cols = [c for c in dtypes.index if c not in index_col_set]
        idx_dtype_vals = [dtypes[c] for c in index_col]
        col_dtype_vals = [dtypes[c] for c in data_cols]
        index_dtypes = pd.Series(idx_dtype_vals, index=index_col)
        dtypes = pd.Series(col_dtype_vals, index=data_cols)

    chunk_bytes = kw.pop("chunk_bytes", None)
    chunk_size = kw.pop("chunk_size", None)

    op = DataFrameReadODPSQuery(
        query=query,
        dtypes=dtypes,
        dtype_backend=dtype_backend,
        string_as_binary=string_as_binary,
        index_columns=index_col,
        index_dtypes=index_dtypes,
        column_renames=col_renames,
        no_split_sql=no_split_sql,
    )
    return op(chunk_bytes=chunk_bytes, chunk_size=chunk_size)
