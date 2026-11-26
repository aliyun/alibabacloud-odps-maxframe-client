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

import abc
import itertools
import threading
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Set, Tuple, Type

if TYPE_CHECKING:
    from maxframe.codegen import DAGCodeGenerator

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

    @classmethod
    def register_operators(cls) -> None:
        """
        Register custom operators and other engine-agnostic registrations.

        Called when the DPE operator tilers are loaded, before
        :meth:`register_dpe_tilers`, so an extension can register the operator
        classes it adds together with the process-scoped support they need —
        optimizer rules, output types, private helpers.
        It runs once per process on the FrameDriver and Ray sides.
        """
        pass

    @classmethod
    def register_dpe_tilers(cls) -> None:
        """
        Register custom DPE operator tilers provided by this extension.

        Called after all built-in DPE tilers have been loaded, so an extension
        can register tilers for its custom operators, e.g. with
        ``register_op_tiler`` from ``maxframe_dpe_engine.tilers.core``.
        """
        pass

    @classmethod
    def register_dpe_executors(cls) -> None:
        """
        Register custom DPE operator executors provided by this extension.

        Called after all built-in DPE executors have been loaded, so an
        extension can register executors for its custom operators, e.g. with
        ``register_op_executor`` from ``maxframe_dpe_execute.executors.core``.
        """
        pass

    @classmethod
    def get_logview_collector_class(cls) -> Optional[Type]:
        """
        Get the logview collector class for this engine.

        Returns
        -------
        Optional[Type]
            The collector class, or None if this engine doesn't provide logview support.
        """
        pass

    @classmethod
    def get_logview_data_field_name(cls) -> Optional[str]:
        """
        Get the field name to store engine-specific logview data in LogviewSessionDetail.

        Returns
        -------
        Optional[str]
            Field name (e.g., 'dpes'), or None if no logview support.
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


class _HookGroupState:
    """Completion state of one hook group.

    ``lock`` serializes the callers that need the same group, so a competing
    thread waits for the registration already running instead of reading a
    half-populated registry.  ``owner`` names the thread running the hooks:
    re-entry from that same thread (an extension looking a registry up while
    registering) must short-circuit rather than recurse.
    """

    __slots__ = ("lock", "owner")

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.owner = None


_registered_hook_groups: Set[Tuple[str, ...]] = set()
_hook_group_states: Dict[Tuple[str, ...], _HookGroupState] = {}
_completed_hooks: Set[Tuple[Tuple[str, str], str]] = set()


def _extension_identity(ext) -> Tuple[str, str]:
    """Provider-stable identity of a loaded extension: (module, qualname).

    Entry-point names are unique only within a distribution, so they cannot
    tell two providers that reuse a name apart; the loaded extension class
    can, and its identity is stable across a retry.
    """
    cls = ext if isinstance(ext, type) else type(ext)
    return cls.__module__, cls.__qualname__


def register_extension_hooks_once(*hook_names: str) -> None:
    """
    Invoke ``hook_names`` on every discovered extension, once per process.

    Registration hooks are triggered lazily from engine dispatch paths, so a
    process may reach them more than once and from more than one thread (DAG
    analysis runs in a worker thread).  Callers of one group are serialized:
    the running thread short-circuits when a hook looks a registry up again
    (no recursion), while a competing thread waits for that registration and
    then either sees the completed key or -- after a failure -- replays the
    group itself.  A failing hook leaves the group incomplete and records each
    ``(extension, hook)`` pair that did succeed, so a retry skips the work
    already done and only replays the extensions that still need it; the
    completed key is only set once every hook succeeded.
    """
    key = tuple(hook_names)
    if key in _registered_hook_groups:
        return
    state = _hook_group_states.setdefault(key, _HookGroupState())
    with state.lock:
        if key in _registered_hook_groups:
            return
        if state.owner == threading.get_ident():
            return
        state.owner = threading.get_ident()
        try:
            for _, ext in iter_extensions():
                ext_id = _extension_identity(ext)
                for hook_name in hook_names:
                    pair = (ext_id, hook_name)
                    if pair in _completed_hooks:
                        continue
                    getattr(ext, hook_name)()
                    _completed_hooks.add(pair)
            # Set while still holding the lock: a waiter that acquires it next
            # must observe the completed group, never rerun the hooks.
            _registered_hook_groups.add(key)
        finally:
            state.owner = None
