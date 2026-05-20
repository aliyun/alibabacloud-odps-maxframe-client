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

import logging
import traceback
from typing import Dict, List

from maxframe.errors import MaxFrameError
from maxframe.lib import wrapped_pickle as pickle
from maxframe.serialization.core import (
    Serializer,
    buffered,
    is_unpickle_allowed,
    pickle_buffers,
    unpickle_buffers,
)
from maxframe.utils import combine_error_message_and_traceback

logger = logging.getLogger(__name__)


class RemoteException(MaxFrameError):
    def __init__(
        self, messages: List[str], tracebacks: List[List[str]], buffers: List[bytes]
    ):
        self.messages = messages
        self.tracebacks = tracebacks
        self.buffers = buffers

    @classmethod
    def from_exception(cls, exc: Exception):
        try:
            buffers = pickle_buffers(exc)
        except:
            if isinstance(exc, (IndexError, KeyError, ValueError)):
                new_exc = type(exc)(*(str(a) for a in exc.args)).with_traceback(
                    exc.__traceback__
                )
                buffers = pickle_buffers(new_exc)
            else:
                logger.exception("Cannot pickle exception %s", exc)
                buffers = []

        messages, tracebacks = [], []
        while exc is not None:
            messages.append(str(exc))
            tracebacks.append(traceback.format_tb(exc.__traceback__))
            exc = exc.__cause__
        return RemoteException(messages, tracebacks, buffers)

    def get_buffers(self) -> List[bytes]:
        return self.buffers

    def get(self) -> Exception:
        return unpickle_buffers(self.buffers) if self.buffers else self

    def __str__(self):
        return combine_error_message_and_traceback(self.messages, self.tracebacks)


class ExceptionSerializer(Serializer):
    @buffered
    def serial(self, obj: Exception, context: Dict):
        if isinstance(obj, RemoteException):
            messages, tracebacks, buffers = obj.messages, obj.tracebacks, obj.buffers
        else:
            remote_exc = RemoteException.from_exception(obj)
            messages, tracebacks = remote_exc.messages, remote_exc.tracebacks
            buffers = remote_exc.get_buffers()
        failure_info = getattr(obj, "_failure_info", None)
        return [messages, tracebacks, failure_info], buffers, True

    def deserial(self, serialized: List, context: Dict, subs: List):
        messages, tracebacks = serialized[:2]
        failure_info = serialized[2] if len(serialized) > 2 else None
        if subs and not pickle.is_unpickle_forbidden() and is_unpickle_allowed():
            try:
                exc = unpickle_buffers(subs)
                if failure_info and not hasattr(exc, "_failure_info"):
                    exc._failure_info = failure_info
                return exc
            except ImportError as ex:
                logger.info(
                    "Failed to load error from module %s, will raise a normal error",
                    ex.name,
                )
        exc = RemoteException(messages, tracebacks, subs)
        if failure_info:
            exc._failure_info = failure_info
        return exc


ExceptionSerializer.register(Exception)
