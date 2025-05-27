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

import os
from typing import Any, BinaryIO, Dict, List, TextIO, Union

from .. import opcodes
from ..core import TILEABLE_TYPE, OutputType
from ..core.operator import ObjectOperator, ObjectOperatorMixin
from ..serialization.serializables import (
    BoolField,
    BytesField,
    DictField,
    Int32Field,
    ListField,
)
from ..typing_ import SessionType, TileableType
from ..utils import to_binary


class RunScript(ObjectOperator, ObjectOperatorMixin):
    _op_type_ = opcodes.RUN_SCRIPT

    code: bytes = BytesField("code", default=None)
    data: Dict[str, TileableType] = DictField("data", default=None)
    retry_when_fail: bool = BoolField("retry_when_fail", default=None)
    command_args: List[str] = ListField("command_args", default=None)
    world_size: int = Int32Field("world_size", default=None)
    rank: int = Int32Field("rank", default=None)

    def __init__(self, command_args=None, **kw):
        command_args = command_args or []
        super().__init__(command_args=command_args, **kw)
        if self.output_types is None:
            self.output_types = [OutputType.object]

    @property
    def retryable(self):
        return self.retry_when_fail

    def has_custom_code(self) -> bool:
        return True

    def __call__(self, inputs):
        return self.new_tileable(inputs)


def _extract_inputs(data: Dict[str, TileableType] = None) -> List[TileableType]:
    if data is not None and not isinstance(data, dict):
        raise TypeError(
            "`data` must be a dict whose key is variable name and value is data"
        )

    inputs = []
    if data is not None:
        for v in data.values():
            if isinstance(v, TILEABLE_TYPE):
                inputs.append(v)

    return inputs


def run_script(
    script: Union[bytes, str, BinaryIO, TextIO],
    data: Dict[str, TileableType] = None,
    n_workers: int = 1,
    command_argv: List[str] = None,
    session: SessionType = None,
    retry_when_fail: bool = True,
    run_kwargs: Dict[str, Any] = None,
):
    """
    Run script in MaxFrame cluster.

    Parameters
    ----------
    script: str or file-like object
        Script to run.
    data: dict
        Variable name to data.
    n_workers: int
        number of workers to run the script
    command_argv: list
        extra command args for script
    session: MaxFrame session
        if not provided, will use default one
    retry_when_fail: bool, default False
       If True, retry when function failed.
    run_kwargs: dict
        extra kwargs for session.run

    Returns
    -------
    Object
        MaxFrame Object.

    """

    if hasattr(script, "read"):
        code = script.read()
    else:
        with open(os.path.abspath(script), "rb") as f:
            code = f.read()

    inputs = _extract_inputs(data)
    op = RunScript(
        data=data,
        code=to_binary(code),
        world_size=n_workers,
        retry_when_fail=retry_when_fail,
        command_args=command_argv,
    )
    return op(inputs).execute(session=session, **(run_kwargs or {}))
