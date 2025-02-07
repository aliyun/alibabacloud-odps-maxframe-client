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
import os
from typing import Any, BinaryIO, Dict, List, Optional, TextIO, Union

from .... import opcodes
from ....remote.run_script import RunScript, _extract_inputs
from ....serialization.serializables import Int32Field, StringField
from ....typing_ import SessionType, TileableType
from ....utils import to_binary

logger = logging.getLogger(__name__)


class RunPyTorch(RunScript):
    _op_type_ = opcodes.RUN_PYTORCH

    # used for chunk op
    master_port = Int32Field("master_port", default=None)
    master_addr = StringField("master_addr", default=None)
    init_method = StringField("init_method", default=None)
    master_waiter_name = StringField("master_waiter_name", default=None)


def run_pytorch_script(
    script: Union[bytes, str, BinaryIO, TextIO],
    n_workers: int,
    data: Dict[str, TileableType] = None,
    gpu: Optional[bool] = None,
    command_argv: List[str] = None,
    retry_when_fail: bool = False,
    session: SessionType = None,
    run_kwargs: Dict[str, Any] = None,
    port: int = None,
    execute: bool = True,
):
    """
    Run PyTorch script in MaxFrame cluster.

    Parameters
    ----------
    script: str or file-like object
        Script to run
    n_workers : int
        Number of PyTorch workers
    data : dict
        Variable name to data.
    gpu : bool
        Run PyTorch script on GPU
    command_argv : list
        Extra command args for script
    retry_when_fail : bool
        If True, retry when function failed.
    session
        MaxFrame session, if not provided, will use default one.
    run_kwargs : dict
        Extra kwargs for `session.run`.
    port : int
        Port of PyTorch worker or ps, will automatically increase for the same worker

    Returns
    -------
    status
        return {'status': 'ok'} if succeeded, or error raised
    """
    if int(n_workers) <= 0:
        raise ValueError("n_workers should be at least 1")
    if hasattr(script, "read"):
        code = script.read()
    else:
        with open(os.path.abspath(script), "rb") as f:
            code = f.read()

    inputs = _extract_inputs(data)
    port = 29500 if port is None else port
    op = RunPyTorch(
        data=data,
        code=to_binary(code),
        world_size=int(n_workers),
        retry_when_fail=retry_when_fail,
        gpu=gpu,
        master_port=port,
        command_args=command_argv,
    )
    t = op(inputs)
    if execute:
        t.execute(session=session, **(run_kwargs or {}))
    else:
        return t
