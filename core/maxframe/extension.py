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

import abc
import itertools
from typing import TYPE_CHECKING, Iterable, Tuple, Type

if TYPE_CHECKING:
    from .codegen import DAGCodeGenerator

    try:
        import maxframe_framedriver.runners
    except ImportError:
        pass


class MaxFrameExtension(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def get_codegen(cls) -> Type["DAGCodeGenerator"]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_runner(cls) -> Type["maxframe_framedriver.runners.SubDagRunner"]:
        raise NotImplementedError

    @classmethod
    async def destroy_session(cls, session_id: str) -> None:
        """
        This interface will be called once a session in the framedriver is ended.

        Parameters
        ----------
        session_id : str
            The session id.
        """
        pass

    @classmethod
    async def create_session(cls, session_id: str) -> None:
        """
        create the session
        Parameters
        ----------
        session_id : str
            The session id.
        """
        pass

    @classmethod
    async def reload_session(cls, session_id: str) -> None:
        """
        Reload the session state when the session is recovered from failover.

        Parameters
        ----------
        session_id : str
            The session id.
        """
        pass

    @classmethod
    def init_service_extension(cls) -> None:
        """
        Init the services of the extension before the app is actually run and will be
        called only once.
        """
        pass

    @classmethod
    def destroy_service_extension(cls) -> None:
        """
        Destroy the services of the extension before the app is actually stopped and
        will be called only once.
        """
        pass


def iter_extensions() -> Iterable[Tuple[str, MaxFrameExtension]]:
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="maxframe.extension")
    except (ImportError, TypeError):
        from importlib_metadata import entry_points

        eps = entry_points(group="maxframe.extension")

    if callable(getattr(eps, "values", None)):
        _it = itertools.chain(*eps.values())
    else:
        _it = eps

    for entry_point in _it:
        yield entry_point.name, entry_point.load()
