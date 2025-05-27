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

import logging
import time
from typing import Any, Dict

import pandas as pd
from odps import ODPS

from .codegen import CodeGenResult
from .codegen.spe.core import SPECodeContext
from .config import option_context
from .core import TileableGraph
from .lib.compat import patch_pandas
from .protocol import ResultInfo
from .typing_ import PandasObjectTypes, TileableType
from .utils import build_temp_table_name

logger = logging.getLogger(__name__)


class DAGCancelledError(Exception):
    pass


class SPEDagRunner:
    def __init__(
        self,
        session_id: str,
        subdag_id: str,
        subdag: TileableGraph,
        generated: CodeGenResult,
        settings: Dict[str, Any],
    ):
        self._session_id = session_id
        self._subdag_id = subdag_id
        self._subdag = subdag
        self._key_to_tileable = {t.key: t for t in subdag}
        self._settings = settings

        with option_context(self._settings) as session_options:
            self._sql_hints = session_options.sql.settings

        self._code = generated.code
        self._constants = generated.constants
        self._out_key_to_vars = generated.output_key_to_variables
        self._out_key_to_infos = generated.output_key_to_result_infos
        self._input_key_to_vars = generated.input_key_to_variables
        self._mark_cancelled = False
        self._odps = ODPS.from_environments()

    def _get_sql_hints(self) -> Dict[str, str]:
        hints = self._sql_hints.copy()
        hints["odps.sql.type.system.odps2"] = True
        return hints

    @staticmethod
    def _pre_process_pandas_data(data: PandasObjectTypes) -> PandasObjectTypes:
        """
        Sort the pandas dataset first to make sure all the tensor can process the
        records in the same order.
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data.sort_index(
                axis=0,
                ascending=True,
                kind="quicksort",
                inplace=True,
                na_position="last",
            )
        elif isinstance(data, pd.Index):
            data.sort_values(return_indexer=False, ascending=True, na_position="last")
        return data

    def fetch_data_by_tileable(self, t: TileableType) -> Any:
        raise NotImplementedError

    def store_data(self, key: str, value: Any) -> ResultInfo:
        raise NotImplementedError

    def run(self) -> Dict[str, ResultInfo]:
        # make forward compatibility of new pandas methods
        patch_pandas()

        local_vars = self._constants
        local_vars[SPECodeContext.logger_var] = logger
        start_timestamp = time.time()
        try:
            # Fetch data
            start_time = time.time()
            for key, var_name in self._input_key_to_vars.items():
                local_vars[var_name] = self.fetch_data_by_tileable(
                    self._key_to_tileable[key]
                )
            logger.info(
                "[%s][%s] fetch data costs: %.3f",
                self._session_id,
                self._subdag_id,
                time.time() - start_time,
            )

            # Execute Python codes
            start_time = time.time()
            logger.info("Generated codes:\n--------\n%s\n--------", self._code)
            exec(self._code, globals(), local_vars)
            logger.info(
                "[%s][%s] execute costs: %.3f",
                self._session_id,
                self._subdag_id,
                time.time() - start_time,
            )

            # Store data
            start_time = time.time()
            result_dict = dict()
            for key, var_name in self._out_key_to_vars.items():
                result_dict[key] = self.store_data(key, local_vars[var_name])
            logger.info(
                "[%s][%s] store data costs: %.3f",
                self._session_id,
                self._subdag_id,
                time.time() - start_time,
            )
            return result_dict
        except Exception as ex:
            local_vars.clear()
            if (
                not isinstance(ex, RuntimeError)
                or not ex.args
                or ex.args[0] != "CANCELLED"
            ):
                raise

            drop_statements = [
                f"DROP TABLE IF EXISTS {build_temp_table_name(self._session_id, key)};"
                for key in self._out_key_to_vars.keys()
            ]
            if drop_statements:  # pragma: no branch
                self._odps.run_sql(
                    "\n".join(drop_statements), hints=self._get_sql_hints()
                )
            raise DAGCancelledError from None
        finally:
            logger.info(
                "[%s][%s] run costs: %.3f",
                self._session_id,
                self._subdag_id,
                time.time() - start_timestamp,
            )

    def mark_cancel(self):
        self._constants["running"] = False
        self._mark_cancelled = True
