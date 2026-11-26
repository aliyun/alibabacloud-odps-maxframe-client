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

import contextlib
import logging
import traceback
from typing import Dict, List, Optional

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


def _log_best_effort(level: int, msg: str, *args) -> None:
    """Log a diagnostic without letting a failing handler escape.

    Deliberately contains only ordinary ``Exception``: a control-flow signal
    (``KeyboardInterrupt``/``SystemExit``) is a process-level event, and these
    helpers are also called outside error-handling blocks, so it must keep
    propagating -- swallowing it would change process semantics, which is worse
    than losing a diagnostic record.
    """
    with contextlib.suppress(Exception):
        logger.log(level, msg, *args)


def try_pickle_buffers(exc: Exception) -> Optional[List[bytes]]:
    """Pickle an exception payload, returning ``None`` if that is not possible.

    Only an ordinary ``Exception`` is contained (an unpicklable payload is not
    an error here). A control-flow signal raised by ``__reduce__`` is
    deliberately left to propagate: this helper is also called outside
    error-handling blocks, so swallowing it would drop a process-level event
    (e.g. a ``Ctrl-C``), which is worse than losing the raw payload.
    """
    try:
        return pickle_buffers(exc)
    except Exception:
        return None


def _format_error_message(exc: BaseException) -> str:
    """Format one exception of the chain without ever raising itself.

    ``str(exc)`` may raise while recording, which would replace the failure
    being recorded with that error, so an ordinary ``Exception`` degrades the
    message to the type name alone. A control-flow signal is deliberately left
    to propagate (see :func:`try_pickle_buffers`): swallowing it would drop a
    process-level event.
    """
    type_name = type(exc).__name__
    try:
        return f"{type_name}: {exc}"
    except Exception:
        _log_best_effort(
            logging.WARNING, "Failed to format the message of %s", type_name
        )
        return type_name


def _format_error_traceback(exc: BaseException) -> List[str]:
    """Format one traceback of the chain without ever raising itself.

    ``traceback.format_tb`` reads source lines through ``linecache``, which can
    fail while recording; an ordinary ``Exception`` degrades to no traceback,
    while a control-flow signal is deliberately left to propagate (see
    :func:`try_pickle_buffers`): swallowing it would drop a process-level event.
    """
    try:
        return traceback.format_tb(exc.__traceback__)
    except Exception:
        _log_best_effort(
            logging.WARNING,
            "Failed to format the traceback of %s",
            type(exc).__name__,
        )
        return []


class RemoteException(MaxFrameError):
    """
    Exception raised on the local side when a remote worker fails.

    Stores the formatted messages and tracebacks of the remote exception chain
    (following ``__cause__``) together with pickled buffers of the original
    exception.

    :meth:`get` reconstructs the original exception from the buffers so it can
    be re-raised on the client side. It doubles as the fallback carrier for
    errors that cannot be transported as the original object, so that the client
    still observes the remote failure and raises this instance instead:

    * unpicklable exceptions, whose ``buffers`` are empty (``get`` then returns
      this instance itself);
    * exceptions that must not be re-raised in the client process, such as
      control-flow signals like ``KeyboardInterrupt`` or ``SystemExit``, which
      are not transported across processes;
    * deserialization denied by the environment (e.g. unpickling forbidden, or
      the originating module unavailable), where clients fall back to this
      instance carrying the same messages and tracebacks.

    Attributes
    ----------
    messages : List[str]
        Error messages of the exception chain, from outermost to innermost, each
        formatted as ``"{type(exc).__name__}: {exc}"`` to keep the original error
        type.
    tracebacks : List[List[str]]
        Formatted traceback lines for each exception in the chain.
    buffers : List[bytes]
        Pickled buffers of the original exception, empty if it cannot be pickled.
    """

    def __init__(
        self, messages: List[str], tracebacks: List[List[str]], buffers: List[bytes]
    ):
        self.messages = messages
        self.tracebacks = tracebacks
        self.buffers = buffers

    @staticmethod
    def _pickle_buffers(exc: Exception) -> List[bytes]:
        """Pickle the exception payload, degrading to no buffers on failure.

        An unpicklable payload must not replace the failure being recorded, so
        the record stays a ``RemoteException``; control-flow signals propagate
        (see :func:`try_pickle_buffers`).
        """
        buffers = try_pickle_buffers(exc)
        if buffers is not None:
            return buffers
        if isinstance(exc, (IndexError, KeyError, ValueError)):
            try:
                buffers = try_pickle_buffers(
                    type(exc)(*(str(a) for a in exc.args)).with_traceback(
                        exc.__traceback__
                    )
                )
            except Exception:
                buffers = None
            if buffers is not None:
                return buffers
        _log_best_effort(
            logging.ERROR, "Cannot pickle exception %s", type(exc).__name__
        )
        return []

    @classmethod
    def from_exception(cls, exc: BaseException):
        if isinstance(exc, RemoteException):
            return exc
        if isinstance(exc, Exception):
            buffers = cls._pickle_buffers(exc)
        else:
            # Errors like KeyboardInterrupt or SystemExit are control-flow
            # signals of the remote process and shall never be re-raised in the
            # client process, hence their payload is not pickled and a
            # RemoteException is raised on the client side instead.
            buffers = []

        messages, tracebacks = [], []
        while exc is not None:
            messages.append(_format_error_message(exc))
            tracebacks.append(_format_error_traceback(exc))
            exc = exc.__cause__
        return RemoteException(messages, tracebacks, buffers)

    def get_buffers(self) -> List[bytes]:
        return self.buffers

    def get(self) -> BaseException:
        if not self.buffers:
            return self
        exc = unpickle_buffers(self.buffers)
        # records written before control-flow signals were filtered may hold a
        # pickled signal: never hand it back for the caller to raise
        return exc if isinstance(exc, Exception) else self

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
                if isinstance(exc, Exception):
                    if failure_info and not hasattr(exc, "_failure_info"):
                        exc._failure_info = failure_info
                    return exc
                # a legacy record may hold a pickled control-flow signal: drop
                # the payload instead of returning an exception that would be
                # re-raised as a signal on this side
                _log_best_effort(
                    logging.WARNING,
                    "Ignoring pickled control-flow signal %s in a serialized error",
                    type(exc).__name__,
                )
                subs = []
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
