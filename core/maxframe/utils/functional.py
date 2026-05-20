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

import dis
import functools
import inspect
import sys
import threading
import types
import warnings
import weakref
from contextlib import contextmanager
from typing import Callable

from maxframe.utils._utils_c import tokenize

_entity_types_cache = None


def _get_entity_types():
    """
    Get combined entity types for isinstance checks.
    Uses lazy import to avoid circular dependencies.

    Returns
    -------
    tuple
        Combined tuple of (Class, DataClass) pairs for all entity types.
    """
    global _entity_types_cache

    if _entity_types_cache is None:
        from maxframe.dataframe.core import (
            DATAFRAME_TYPE,
            GROUPBY_TYPE,
            INDEX_TYPE,
            SERIES_TYPE,
        )

        _entity_types_cache = DATAFRAME_TYPE + SERIES_TYPE + INDEX_TYPE + GROUPBY_TYPE

    return _entity_types_cache


def _check_partial(func, entities):
    """
    Check functools.partial for entities in args/kwargs.
    """
    while isinstance(func, functools.partial):
        # Check positional arguments
        for arg in func.args:
            if isinstance(arg, _get_entity_types()):
                entities.append(arg)

        # Check keyword arguments
        for value in func.keywords.values():
            if isinstance(value, _get_entity_types()):
                entities.append(value)

        # Recursively check wrapped function
        func = func.func
    return func


def _check_closure(func, entities):
    """
    Check closure variables for entities.
    """
    if hasattr(func, "__closure__") and func.__closure__:
        for cell in func.__closure__:
            try:
                value = cell.cell_contents
                if isinstance(value, _get_entity_types()):
                    entities.append(value)
                # Recursively check nested functions
                elif callable(value):
                    _check_closure_for_entities_cached(value)  # noqa: F821
            except ValueError:
                # Cell is empty
                pass


def _check_globals(func, entities):
    """
    Check globals referenced by function via bytecode.
    """

    if not hasattr(func, "__globals__") or not hasattr(func, "__code__"):
        return

    # Pre-filter: check if globals contain any entities
    entity_globals = {
        name: obj
        for name, obj in func.__globals__.items()
        if isinstance(obj, _get_entity_types())
    }

    if not entity_globals:
        return  # No entities in globals, skip bytecode inspection

    # Get referenced global names via bytecode
    referenced_names = set()
    for instruction in dis.get_instructions(func.__code__):
        if instruction.opname == "LOAD_GLOBAL":
            referenced_names.add(instruction.argval)

    # Check if referenced globals are entities
    for name in referenced_names:
        if name in entity_globals:
            entities.append(entity_globals[name])


@functools.lru_cache(maxsize=128)
def _check_closure_for_entities_cached(func):
    """
    Cached internal inspection function.
    """
    entities = []

    func = unwrap_function(_check_partial(func, entities))
    _check_closure(func, entities)
    _check_globals(func, entities)
    return entities


def check_closure_for_entities(func, operation_name="apply"):
    """
    Check if a function's closure, globals, or partial args contain MaxFrame entity objects.

    Parameters
    ----------
    func : callable
        The function to inspect.
    operation_name : str, optional
        Name of the operation for warning message (e.g., "apply", "apply_chunk").
        Default is "apply".

    Raises
    ------
    UserWarning
        If MaxFrame entities are found in closure, globals, or partial args.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'a': [1, 2, 3]})
    >>> def func(x):
    ...     return df['a'] + x  # df is captured in closure
    >>> check_closure_for_entities(func, "apply")
    UserWarning: MaxFrame entities found in function for apply. This may cause
    unexpected behavior during distributed execution.
    """
    entities = _check_closure_for_entities_cached(func)
    if entities:
        warnings.warn(
            f"MaxFrame entities found in function for {operation_name}. "
            "This may cause unexpected behavior during distributed execution. "
            "Try using md.merge instead or execute and fetch the entity first "
            "and reference local object instead.",
            UserWarning,
            stacklevel=3,
        )


def implements(f: Callable):
    """
    Decorator to copy documentation from one function to another.

    Parameters
    ----------
    f : callable
        The function to copy documentation from.

    Returns
    -------
    decorator : callable
        The decorator function.
    """

    def decorator(g):
        g.__doc__ = f.__doc__
        return g

    return decorator


def skip_na_call(func: Callable):
    """
    Decorator that skips calling the function if the argument is None.

    Parameters
    ----------
    func : callable
        The function to wrap.

    Returns
    -------
    wrapper : callable
        The wrapped function.
    """

    @functools.wraps(func)
    def new_func(x):
        return func(x) if x is not None else None

    return new_func


_enter_counter = 0
_initial_session = None


def enter_current_session(func: Callable):
    """
    Decorator that manages session context.

    Parameters
    ----------
    func : callable
        The function to wrap.

    Returns
    -------
    wrapper : callable
        The wrapped function.
    """

    @functools.wraps(func)
    def wrapped(cls, ctx, op):
        from maxframe.session import AbstractSession, get_default_session

        global _enter_counter, _initial_session
        # skip in some test cases
        if not hasattr(ctx, "get_current_session"):
            return func(cls, ctx, op)

        with AbstractSession._lock:
            if _enter_counter == 0:
                # to handle nested call, only set initial session
                # in first call
                session = ctx.get_current_session()
                _initial_session = get_default_session()
                session.as_default()
            _enter_counter += 1

        try:
            result = func(cls, ctx, op)
        finally:
            with AbstractSession._lock:
                _enter_counter -= 1
                if _enter_counter == 0:
                    # set previous session when counter is 0
                    if _initial_session:
                        _initial_session.as_default()
                    else:
                        AbstractSession.reset_default()
        return result

    return wrapped


def ignore_warning(func: Callable):
    """
    Decorator that catches and ignores warnings.

    Parameters
    ----------
    func : callable
        The function to wrap.

    Returns
    -------
    wrapper : callable
        The wrapped function.
    """

    @functools.wraps(func)
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return inner


def deprecate_positional_args(func_or_version=None):
    """
    Decorator to deprecate positional arguments for a function.

    When a function signature changes from `func(a, b=1, c=2, **kw)` to
    `func(a, *, b=1, c=2, **kw)`, this decorator ensures backward compatibility
    while issuing a warning when positional arguments are used in place of
    keyword-only arguments.

    Parameters
    ----------
    func_or_version : callable or str, optional
        Either the function to decorate or a version string for the deprecation.

    Returns
    -------
    decorator or wrapper : callable
        Either the decorator function or the wrapped function.
    """

    def _make_deprecate_wrapper(func, version):
        """Helper function to create the actual wrapper with deprecation logic."""
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        # Count how many parameters can be positional (before keyword-only parameters)
        # KEYWORD_ONLY parameters are those that come after '*' in the function signature
        max_positional_args = 0
        for param_name in param_names:
            param = sig.parameters[param_name]
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                max_positional_args += 1
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                # Once we encounter a keyword-only parameter, we stop counting
                # as all subsequent parameters must be keyword-only
                break

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) <= max_positional_args:
                return func(*args, **kwargs)

            new_kwargs = kwargs.copy()

            # Some args are being passed positionally that should be keyword-only
            invalid_params = []
            for i, arg in enumerate(args[max_positional_args:]):
                if i + max_positional_args < len(param_names):
                    param_name = param_names[i + max_positional_args]
                    invalid_params.append(param_name)
                    new_kwargs[param_name] = arg

            # Issue a single warning with all invalid parameter names
            if invalid_params:
                version_str = f" in version {version}" if version else ""
                params_str = ", ".join(f"'{param}'" for param in invalid_params)
                warnings.warn(
                    f"Passing {params_str} as positional argument(s) is "
                    f"deprecated{version_str}. Please pass these as keyword "
                    f"argument(s) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            return func(*args[:max_positional_args], **new_kwargs)

        return wrapper

    if callable(func_or_version):
        func = func_or_version
        return _make_deprecate_wrapper(func, None)
    else:
        version = func_or_version

        def decorator(func):
            return _make_deprecate_wrapper(func, version)

        return decorator


def unwrap_function(func, stop_predicate: Callable = None):
    unwrapped_func = func
    while True:
        if stop_predicate is not None and stop_predicate(unwrapped_func):
            break
        if isinstance(unwrapped_func, functools.partial):
            unwrapped_func = unwrapped_func.func
        elif hasattr(unwrapped_func, "__wrapped__"):
            unwrapped_func = unwrapped_func.__wrapped__
        else:
            break
    return unwrapped_func


_func_token_cache = weakref.WeakKeyDictionary()


def _get_func_token_values(func):
    if hasattr(func, "__code__"):
        tokens = [func.__code__.co_code]
        if func.__closure__ is not None:
            cvars = tuple(x.cell_contents for x in func.__closure__)
            tokens.append(cvars)
        return tokens
    else:
        tokens = []
        while isinstance(func, functools.partial):
            tokens.extend([func.args, func.keywords])
            func = func.func
        if hasattr(func, "__code__"):
            tokens.extend(_get_func_token_values(func))
        elif isinstance(func, types.BuiltinFunctionType):
            tokens.extend([func.__module__, func.__qualname__])
        else:
            tokens.append(func)
        return tokens


def get_func_token(func):
    try:
        token = _func_token_cache.get(func)
        if token is None:
            fields = _get_func_token_values(func)
            token = tokenize(*fields)
            _func_token_cache[func] = token
        return token
    except TypeError:  # cannot create weak reference to func like 'numpy.ufunc'
        return tokenize(*_get_func_token_values(func))


_io_quiet_local = threading.local()
_io_quiet_lock = threading.Lock()


class _QuietIOWrapper:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getattr__(self, item):
        return getattr(self.wrapped, item)

    def write(self, d):
        if getattr(_io_quiet_local, "is_wrapped", False):
            return 0
        return self.wrapped.write(d)


@contextmanager
def quiet_stdio():
    """Quiets standard outputs when inferring types of functions"""
    with _io_quiet_lock:
        _io_quiet_local.is_wrapped = True
        sys.stdout = _QuietIOWrapper(sys.stdout)
        sys.stderr = _QuietIOWrapper(sys.stderr)

    try:
        yield
    finally:
        with _io_quiet_lock:
            sys.stdout = sys.stdout.wrapped
            sys.stderr = sys.stderr.wrapped
            if not isinstance(sys.stdout, _QuietIOWrapper):
                _io_quiet_local.is_wrapped = False
