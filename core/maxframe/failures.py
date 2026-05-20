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

import warnings
from typing import Optional

import pandas as pd
from odps import ODPS

from maxframe.core import OutputType
from maxframe.errors import get_failure_info_from_exception
from maxframe.io.odpsio import ODPSTableIO
from maxframe.io.odpsio.arrow import arrow_to_pandas
from maxframe.io.odpsio.schema import arrow_table_to_pandas_dataframe
from maxframe.protocol import DataFrameTableMeta
from maxframe.utils import sync_pyodps_options


class FailureInfo:
    __slots__ = (
        "substep_id",
        "operator_name",
        "error_message",
        "table",
        "table_meta",
        "_cached_data",
    )

    def __init__(
        self,
        substep_id: Optional[str] = None,
        operator_name: Optional[str] = None,
        error_message: Optional[str] = None,
        table: Optional[str] = None,
        table_meta: Optional[DataFrameTableMeta] = None,
    ):
        self.substep_id = substep_id
        self.operator_name = operator_name
        self.error_message = error_message
        self.table = table
        self.table_meta = table_meta
        self._cached_data = None

    def hydrate(self, odps_entry=None) -> "FailureInfo":
        if self.table:
            try:
                self.fetch_data(odps_entry=odps_entry)
            except Exception:
                warnings.warn(
                    "Failed to hydrate FailureInfo from table",
                    RuntimeWarning,
                )
        return self

    def fetch_data(self, odps_entry=None):
        if self._cached_data is not None:
            return self._cached_data

        if self.table:
            if odps_entry is None:
                try:
                    odps_entry = ODPS.from_environments()
                except Exception as exc:
                    raise ValueError(
                        "Cannot fetch failure data: pass an odps_entry argument."
                    ) from exc

            with sync_pyodps_options():
                table_io = ODPSTableIO(odps_entry)
                with table_io.open_reader(self.table) as reader:
                    arrow_table = reader.read_all()
            if self.table_meta is not None:
                failed_rows = arrow_to_pandas(arrow_table, self.table_meta)
            else:
                failed_rows = arrow_table_to_pandas_dataframe(arrow_table)
            self._cached_data = failed_rows
            return failed_rows

        return None

    @property
    def has_table(self) -> bool:
        return bool(self.table)

    def __repr__(self) -> str:
        cls = type(self).__name__
        parts = []
        if self.operator_name is not None:
            parts.append(f"operator_name={self.operator_name!r}")
        if self.substep_id is not None:
            parts.append(f"substep_id={self.substep_id!r}")
        if self.error_message is not None:
            parts.append(f"error_message={self.error_message!r}")
        if self.table is not None:
            parts.append(f"table={self.table!r}")
        if not parts:
            return f"{cls}()"
        return f"{cls}({', '.join(parts)})"


def _load_table_meta(raw_table_meta) -> Optional[DataFrameTableMeta]:
    if isinstance(raw_table_meta, DataFrameTableMeta):
        return raw_table_meta
    if not isinstance(raw_table_meta, dict):
        return None

    if isinstance(raw_table_meta.get("pd_column_dtypes"), pd.Series):
        try:
            table_meta = raw_table_meta.copy()
            meta_type = table_meta.get("type")
            if not isinstance(meta_type, OutputType):
                table_meta["type"] = OutputType(meta_type)
            return DataFrameTableMeta(**table_meta)
        except Exception:
            warnings.warn(
                "Failed to construct DataFrameTableMeta from dict",
                RuntimeWarning,
            )
            return None

    try:
        return DataFrameTableMeta.from_json(raw_table_meta)
    except Exception:
        warnings.warn(
            "Failed to deserialize DataFrameTableMeta",
            RuntimeWarning,
        )
        return None


def get_failure_info(
    exception: BaseException, odps_entry: Optional[ODPS] = None
) -> Optional[FailureInfo]:
    raw = get_failure_info_from_exception(exception)
    if not raw:
        return None
    info = FailureInfo(
        substep_id=raw.get("substep_id"),
        operator_name=raw.get("operator_name"),
        error_message=str(exception),
        table=raw.get("table"),
        table_meta=_load_table_meta(raw.get("table_meta")),
    )
    if odps_entry is not None:
        info.hydrate(odps_entry=odps_entry)
    return info
