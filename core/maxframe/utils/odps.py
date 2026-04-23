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

import datetime
import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

from odps import ODPS
from odps.config import option_context as pyodps_option_context
from odps.models import Project, Table, TableSchema
from odps.models.schema import Schema

logger = logging.getLogger(__name__)


def build_temp_table_name(session_id: str, tileable_key: str) -> str:
    return f"tmp_mf_{session_id}_{tileable_key}"


def build_temp_intermediate_table_name(session_id: str, tileable_key: str) -> str:
    temp_table = build_temp_table_name(session_id, tileable_key)
    return f"{temp_table}_intermediate"


def build_session_volume_name(session_id: str) -> str:
    return f"mf_vol_{session_id.replace('-', '_')}"


@contextmanager
def sync_pyodps_options():
    from odps.config import option_context as pyodps_option_context

    from ..config import options

    with pyodps_option_context() as cfg:
        cfg.local_timezone = options.local_timezone
        if options.session.enable_schema:
            cfg.enable_schema = options.session.enable_schema
        yield


def update_wlm_quota_settings(session_id: str, engine_settings: Dict[str, Any]):
    from ..config import options

    engine_quota = engine_settings.get("odps.task.wlm.quota", None)
    session_quota = options.session.quota_name or None
    if engine_quota != session_quota and engine_quota:
        logger.warning(
            "[Session=%s] Session quota (%s) is different to SubDag engine quota (%s)",
            session_id,
            session_quota,
            engine_quota,
        )
        raise ValueError(
            "Quota name cannot be changed after sessions are created, "
            f"session_quota={session_quota}, engine_quota={engine_quota}"
        )

    if session_quota:
        engine_settings["odps.task.wlm.quota"] = session_quota
    elif "odps.task.wlm.quota" in engine_settings:
        engine_settings.pop("odps.task.wlm.quota")


def get_default_table_properties():
    return {"storagestrategy": "archive"}


def config_odps_default_options():
    from odps import options as odps_options

    odps_options.sql.settings = {
        "odps.longtime.instance": "false",
        "odps.sql.session.select.only": "false",
        "metaservice.client.cache.enable": "false",
        "odps.sql.session.result.cache.enable": "false",
        "odps.sql.submit.mode": "script",
        "odps.sql.job.max.time.hours": 72,
    }


def get_odps_dlf_table(
    odps_entry: ODPS,
    table_name: str,
    schema: Optional[str] = None,
    project: Optional[str] = None,
    sql_hints: Optional[Dict[str, str]] = None,
    inst_logger: Optional[logging.Logger] = None,
):
    def adapt_dlf_schema(src_schema_list):
        ret = []
        for c in src_schema_list:
            col_data = {}
            for k, v in c.items():
                k = k.lower()
                if k == "nullable":
                    k = "isNullable"
                if v in ("true", "false"):
                    v = v == "true"
                col_data[k] = v
            ret.append(col_data)
        return ret

    inst_logger = inst_logger or logger

    if hasattr(odps_entry, "get_default_project_name"):
        project = project or odps_entry.get_default_project_name()
    else:
        project = project or odps_entry.project

    schema = schema or "default"
    if "." in table_name:
        full_table_name = table_name
        name_parts = table_name.split(".")
        if len(name_parts) == 2:
            schema, table_name = name_parts
        else:
            assert len(name_parts) == 3
            project, schema, table_name = name_parts
        if table_name.startswith("`") and table_name.endswith("`"):
            table_name = table_name[1:-1]
    else:
        full_table_name = ".".join([project, schema, f"`{table_name}`"])

    ddl_sql_hints = (sql_hints or {}).copy()
    ddl_sql_hints["odps.sql.submit.mode"] = ""
    ddl_sql_hints["odps.sql.session.select.only"] = ""
    ddl_sql_hints["odps.namespace.schema"] = "true"
    ddl_sql_hints["odps.sql.select.output.format"] = "json"

    run_sql_offline = getattr(odps_entry, "run_sql_offline", None) or getattr(
        odps_entry, "run_sql", None
    )
    assert run_sql_offline
    with pyodps_option_context() as pyodps_options:
        pyodps_options.sql.settings = ddl_sql_hints
        inst = run_sql_offline(f"DESC EXTENDED {full_table_name}", hints=ddl_sql_hints)

    inst_logger.info(
        "Resolving schema for table %s with instance %s",
        full_table_name,
        inst.id,
    )
    inst.wait_for_success()

    schema_json = json.loads(inst.get_task_result())
    inst_logger.info(
        "Result of resolved schema of table %s: %s",
        full_table_name,
        schema_json,
    )
    if "NativeColumns" in schema_json:
        schema_json["columns"] = adapt_dlf_schema(schema_json.pop("NativeColumns"))
    if "PartitionColumns" in schema_json:
        schema_json["partitionKeys"] = adapt_dlf_schema(
            schema_json.pop("PartitionColumns")
        )
    else:
        schema_json["partitionKeys"] = []

    # generate mock parent objects
    project_obj = Project(name=project)
    schema_obj = Schema(name=schema, parent=project_obj)

    table_obj = Table(name=table_name, parent=schema_obj)
    tb_schema_obj = TableSchema(parent=table_obj)
    tb_schema_obj = tb_schema_obj.parse(None, schema_json, obj=tb_schema_obj)
    tb_schema_obj.load()

    table_obj.table_schema = tb_schema_obj
    table_obj.type = Table.Type.EXTERNAL_TABLE
    table_obj.last_data_modified_time = datetime.datetime.now()
    return table_obj


_survey_logs = []


def add_survey_log(survey_data: dict):
    _survey_logs.append(survey_data)


def submit_survey_logs(odps_entry: ODPS):
    global _survey_logs
    cur_logs, _survey_logs = _survey_logs, []
    log_url = "/".join([odps_entry.get_project().resource(), "logs"])
    dw_environs = {k: v for k, v in os.environ.items() if k.startswith("SKYNET_")}
    try:
        for log in cur_logs:
            log = log.copy()
            log.update(dw_environs)
            odps_entry.rest.put(log_url, json.dumps(log))
    except:
        pass
