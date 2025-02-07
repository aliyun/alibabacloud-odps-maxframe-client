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

import io
from typing import Any, Callable, Dict

from ....core import TILEABLE_TYPE
from ....lib import wrapped_pickle as pickle
from ....typing_ import SessionType
from ....utils import find_objects, replace_objects
from .run_script import run_pytorch_script

_script_template = """
import cloudpickle
from maxframe.utils import replace_objects


def main(**kwargs):
    func = cloudpickle.loads(%(pickled_func)r)
    nested = cloudpickle.loads(%(nested)r)
    args, kw = replace_objects(nested, kwargs)
    return func(*args, **kw)


if __name__ == "__main__":
    vars = dict()
    for var_name in cloudpickle.loads(%(var_names)r):
        vars[var_name] = globals()[var_name]
    main(**vars)
"""


def run_pytorch_function(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    *,
    n_workers: int = 1,
    retry_when_fail: bool = False,
    session: SessionType = None,
    run_kwargs: Dict[str, Any] = None,
    port: int = None,
    execute: bool = True,
):
    """
    Run PyTorch function in MaxFrame cluster.

    Besides args and kwargs, the function will receive extra
    environment variables from the caller:

    * MASTER_ADDR, MASTER_PORT: the endpoint of the master node
    * RANK: the index of the current worker

    Parameters
    ----------
    func: Callable
        Function or callable object to run
    args: tuple
        Args tuple to pass to the function
    kwargs: dict
    n_workers : int
        Number of PyTorch workers
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
    kwargs = kwargs or {}
    packed = [args, kwargs]
    inputs = find_objects(packed, TILEABLE_TYPE)
    input_to_var = {t: f"torch_var_{i}" for i, t in enumerate(inputs)}
    var_to_input = {v: k for k, v in input_to_var.items()}
    structure = replace_objects(packed, input_to_var)

    replaces = {
        "pickled_func": pickle.dumps(func),
        "nested": pickle.dumps(structure),
        "var_names": pickle.dumps(list(input_to_var.values())),
    }
    code = _script_template % replaces
    return run_pytorch_script(
        io.StringIO(code),
        n_workers,
        data=var_to_input,
        retry_when_fail=retry_when_fail,
        session=session,
        run_kwargs=run_kwargs,
        port=port,
        execute=execute,
    )
