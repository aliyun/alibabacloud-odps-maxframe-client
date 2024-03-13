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

import dataclasses
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from odps import ODPS
from odps.types import Column, OdpsSchema, validate_data_type

from ... import opcodes
from ...core import OutputType
from ...core.graph import DAG
from ...odpsio import odps_schema_to_pandas_dtypes
from ...serialization.serializables import (
    AnyField,
    BoolField,
    FieldTypes,
    Int64Field,
    ListField,
    SeriesField,
    StringField,
)
from ..utils import parse_index
from .core import ColumnPruneSupportedDataSourceMixin, IncrementalIndexDatasource

_EXPLAIN_DEPENDS_REGEX = re.compile(r"([^\s]+) depends on: ([^\n]+)")
_EXPLAIN_JOB_REGEX = re.compile(r"(\S+) is root job")
_EXPLAIN_TASKS_HEADER_REGEX = re.compile(r"In Job ([^:]+):")
_EXPLAIN_ROOT_TASKS_REGEX = re.compile(r"root Tasks: (.+)")
_EXPLAIN_TASK_REGEX = re.compile(r"In Task ([^:]+)")
_EXPLAIN_TASK_SCHEMA_REGEX = re.compile(
    r"In Task ([^:]+)[\S\s]+FS: output: ([^\n #]+)[\s\S]+schema:\s+([\S\s]+)$",
    re.MULTILINE,
)
_EXPLAIN_COLUMN_REGEX = re.compile(r"([^ ]+) \(([^)]+)\)(?:| AS ([^ ]+))(?:\n|$)")


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
        if grouped and not part.startswith(" "):
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


def _parse_explained_schema(explain_string: str) -> OdpsSchema:
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
    if len(indep_job_names) > 1:  # pragma: no cover
        raise ValueError("Only one final job is allowed in SQL statement")

    tasks_sector = jobs_sector.jobs[indep_job_names[0]]
    task_dag = tasks_sector.build_dag()
    indep_task_names = list(task_dag.iter_indep(reverse=True))
    if len(indep_task_names) > 1:  # pragma: no cover
        raise ValueError("Only one final task is allowed in SQL statement")

    task_sector = tasks_sector.tasks[indep_task_names[0]]
    if not task_sector.schema:  # pragma: no cover
        raise ValueError("Cannot detect output schema")
    if task_sector.output_target != "Screen":
        raise ValueError("The SQL statement should be an instant query")
    cols = [
        Column(c.column_alias or c.column_name, validate_data_type(c.column_type))
        for c in task_sector.schema
    ]
    return OdpsSchema(cols)


class DataFrameReadODPSQuery(
    IncrementalIndexDatasource,
    ColumnPruneSupportedDataSourceMixin,
):
    _op_type_ = opcodes.READ_ODPS_QUERY

    query = StringField("query")
    dtypes = SeriesField("dtypes", default=None)
    columns = AnyField("columns", default=None)
    nrows = Int64Field("nrows", default=None)
    use_arrow_dtype = BoolField("use_arrow_dtype", default=None)
    string_as_binary = BoolField("string_as_binary", default=None)
    index_columns = ListField("index_columns", FieldTypes.string, default=None)
    index_dtypes = SeriesField("index_dtypes", default=None)

    def get_columns(self):
        return self.columns

    def set_pruned_columns(self, columns, *, keep_order=None):  # pragma: no cover
        self.columns = columns

    def __call__(self, chunk_bytes=None, chunk_size=None):
        if not self.index_columns:
            index_value = parse_index(pd.RangeIndex(0))
        elif len(self.index_columns) == 1:
            index_value = parse_index(
                pd.Index([], name=self.index_columns[0]).astype(self.index_dtypes[0])
            )
        else:
            idx = pd.MultiIndex.from_frame(
                pd.DataFrame([], columns=self.index_columns).astype(self.index_dtypes)
            )
            index_value = parse_index(idx)

        columns_value = parse_index(self.dtypes.index, store_data=True)
        self.output_types = [OutputType.dataframe]
        return self.new_tileable(
            [],
            None,
            shape=(len(self.dtypes), np.nan),
            dtypes=self.dtypes,
            index_value=index_value,
            columns_value=columns_value,
            chunk_bytes=chunk_bytes,
            chunk_size=chunk_size,
        )


def read_odps_query(
    query: str,
    odps_entry: ODPS = None,
    index_col: Union[None, str, List[str]] = None,
    string_as_binary: bool = None,
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

    Returns
    -------
    result: DataFrame
        DataFrame read from MaxCompute (ODPS) table
    """
    odps_entry = odps_entry or ODPS.from_environments()
    inst = odps_entry.execute_sql(f"EXPLAIN {query}")
    explain_str = list(inst.get_task_results().values())[0]

    odps_schema = _parse_explained_schema(explain_str)
    dtypes = odps_schema_to_pandas_dtypes(odps_schema)

    if not index_col:
        index_dtypes = None
    else:
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
        use_arrow_dtype=kw.pop("use_arrow_dtype", True),
        string_as_binary=string_as_binary,
        index_columns=index_col,
        index_dtypes=index_dtypes,
    )
    return op(chunk_bytes=chunk_bytes, chunk_size=chunk_size)
