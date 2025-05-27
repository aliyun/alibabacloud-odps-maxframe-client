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

from abc import ABC, abstractmethod
from typing import Any, List

from ..utils import classproperty


class Context(ABC):
    """
    Context that providing API that can be
    used inside `tile` and `execute`.
    """

    all_contexts = []

    @abstractmethod
    def get_session_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_subdag_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_local_host_ip(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_worker_cores(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_chunks_result(self, data_keys: List[str], copy: bool = True) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def create_operator_controller(self, name: str, object_cls, *args, **kwargs):
        """
        Create operator controller.

        Parameters
        ----------
        name : str
            Object name.
        object_cls
            Object class.
        args
        kwargs

        Returns
        -------
        ref
        """

    @abstractmethod
    def get_operator_controller(self, name: str):
        """
        Get remote object

        Parameters
        ----------
        name : str
            Object name.

        Returns
        -------
        ref
        """

    @abstractmethod
    def destroy_operator_controller(self, name: str):
        """
        Destroy remote object.

        Parameters
        ----------
        name : str
            Object name.
        """

    def __enter__(self):
        Context.all_contexts.append(self)

    def __exit__(self, *_):
        Context.all_contexts.pop()

    @classproperty
    def current(cls):
        return cls.all_contexts[-1] if cls.all_contexts else None


def set_context(context: Context):
    Context.all_contexts.append(context)


def get_context() -> Context:
    return Context.current
