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

import asyncio.events
import concurrent.futures
import contextlib
import contextvars
import copy
import dataclasses
import datetime
import enum
import functools
import importlib
import inspect
import io
import itertools
import logging
import math
import numbers
import os
import pkgutil
import random
import re
import struct
import sys
import tempfile
import threading
import time
import tokenize as pytokenize
import types
import warnings
import weakref
import zlib
from collections.abc import Hashable, Mapping
from contextlib import contextmanager
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import msgpack
import numpy as np
import pandas as pd
import traitlets
from tornado import httpclient, web
from tornado.simple_httpclient import HTTPTimeoutError

from ._utils import (  # noqa: F401 # pylint: disable=unused-import
    NamedType,
    Timer,
    TypeDispatcher,
    ceildiv,
    get_user_call_point,
    new_random_id,
    register_tokenizer,
    reset_id_random_seed,
    to_binary,
    to_str,
    to_text,
    tokenize,
    tokenize_int,
)
from .lib.dtypes_extension import ArrowDtype
from .lib.version import parse as parse_version
from .typing_ import TileableType, TimeoutType

# make flake8 happy by referencing these imports
NamedType = NamedType
TypeDispatcher = TypeDispatcher
tokenize = tokenize
register_tokenizer = register_tokenizer
ceildiv = ceildiv
reset_id_random_seed = reset_id_random_seed
new_random_id = new_random_id
get_user_call_point = get_user_call_point
_is_ci = (os.environ.get("CI") or "0").lower() in ("1", "true")
pd_release_version: Tuple[int] = parse_version(pd.__version__).release

logger = logging.getLogger(__name__)

try:
    from pandas._libs import lib as _pd__libs_lib
    from pandas._libs.lib import NoDefault, no_default

    _raw__reduce__ = type(NoDefault).__reduce__

    def _no_default__reduce__(self):
        if self is not NoDefault:
            return _raw__reduce__(self)
        else:  # pragma: no cover
            return getattr, (_pd__libs_lib, "NoDefault")

    if hasattr(_pd__libs_lib, "_NoDefault"):  # pragma: no cover
        # need to patch __reduce__ to make sure it can be properly unpickled
        type(NoDefault).__reduce__ = _no_default__reduce__
    else:
        # introduced in pandas 1.5.0 : register for pickle compatibility
        _pd__libs_lib._NoDefault = NoDefault
except ImportError:  # pragma: no cover

    class NoDefault(enum.Enum):
        no_default = "NO_DEFAULT"

        def __repr__(self) -> str:
            return "<no_default>"

    no_default = NoDefault.no_default

    try:
        # register for pickle compatibility
        from pandas._libs import lib as _pd__libs_lib

        _pd__libs_lib.NoDefault = NoDefault
    except (ImportError, AttributeError):
        pass

try:
    import pyarrow as pa
except ImportError:
    pa = None

try:
    from pandas import ArrowDtype as PandasArrowDtype  # noqa: F401

    ARROW_DTYPE_NOT_SUPPORTED = False
except ImportError:
    ARROW_DTYPE_NOT_SUPPORTED = True


class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


def implements(f: Callable):
    def decorator(g):
        g.__doc__ = f.__doc__
        return g

    return decorator


class AttributeDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttributeDict' object has no attribute {item}")


def on_serialize_shape(shape: Tuple[int]):
    def _to_shape_num(x):
        if np.isnan(x):
            return -1
        if isinstance(x, np.generic):
            return x.item()
        return x

    if shape:
        return tuple(_to_shape_num(s) for s in shape)
    return shape


def on_deserialize_shape(shape: Tuple[int]):
    if shape:
        return tuple(s if s != -1 else np.nan for s in shape)
    return shape


def on_serialize_numpy_type(value: np.dtype):
    if value is pd.NaT:
        value = None
    return value.item() if isinstance(value, np.generic) else value


def on_serialize_nsplits(value: Tuple[Tuple[int]]):
    if value is None:
        return None
    new_nsplits = []
    for dim_splits in value:
        new_nsplits.append(tuple(None if pd.isna(v) else v for v in dim_splits))
    return tuple(new_nsplits)


def has_unknown_shape(
    *tiled_tileables: TileableType, axis: Union[None, int, List[int]] = None
) -> bool:
    if isinstance(axis, int):
        axis = [axis]

    for tileable in tiled_tileables:
        if getattr(tileable, "shape", None) is None:
            continue

        shape_iter = (
            tileable.shape if axis is None else (tileable.shape[idx] for idx in axis)
        )
        if any(pd.isnull(s) for s in shape_iter):
            return True

        nsplits_iter = (
            tileable.nsplits
            if axis is None
            else (tileable.nsplits[idx] for idx in axis)
        )
        if any(pd.isnull(s) for s in itertools.chain(*nsplits_iter)):
            return True
    return False


def calc_nsplits(chunk_idx_to_shape: Dict[Tuple[int], Tuple[int]]) -> Tuple[Tuple[int]]:
    """
    Calculate a tiled entity's nsplits.

    Parameters
    ----------
    chunk_idx_to_shape : Dict type, {chunk_idx: chunk_shape}

    Returns
    -------
    nsplits
    """
    ndim = len(next(iter(chunk_idx_to_shape)))
    tileable_nsplits = []
    # for each dimension, record chunk shape whose index is zero on other dimensions
    for i in range(ndim):
        splits = []
        for index, shape in chunk_idx_to_shape.items():
            if all(idx == 0 for j, idx in enumerate(index) if j != i):
                splits.append(shape[i])
        tileable_nsplits.append(tuple(splits))
    return tuple(tileable_nsplits)


def copy_tileables(tileables: List[TileableType], **kwargs):
    inputs = kwargs.pop("inputs", None)
    copy_key = kwargs.pop("copy_key", True)
    copy_id = kwargs.pop("copy_id", True)
    if kwargs:
        raise TypeError(f"got un unexpected keyword argument '{next(iter(kwargs))}'")
    if len(tileables) > 1:
        # cannot handle tileables with different operators here
        # try to copy separately if so
        if len({t.op for t in tileables}) != 1:
            raise TypeError("All tileables' operators should be same.")

    op = tileables[0].op.copy().reset_key()
    if copy_key:
        op._key = tileables[0].op.key
    kws = []
    for t in tileables:
        params = t.params.copy()
        if copy_key:
            params["_key"] = t.key
        if copy_id:
            params["_id"] = t.id
        params.update(t.extra_params)
        kws.append(params)
    inputs = inputs or op.inputs
    return op.new_tileables(inputs, kws=kws, output_limit=len(kws))


def make_dtype(dtype: Union[np.dtype, pd.api.extensions.ExtensionDtype]):
    if dtype is None:
        return None
    elif (
        isinstance(dtype, str) and dtype == "category"
    ) or pd.api.types.is_extension_array_dtype(dtype):
        # return string dtype directly as legacy python version
        #  does not support ExtensionDtype
        return dtype
    elif dtype is pd.Timestamp or dtype is datetime.datetime:
        return np.dtype("datetime64[ns]")
    elif dtype is pd.Timedelta or dtype is datetime.timedelta:
        return np.dtype("timedelta64[ns]")
    else:
        try:
            return pd.api.types.pandas_dtype(dtype)
        except TypeError:
            return np.dtype("O")


def make_dtypes(
    dtypes: Union[
        list, dict, str, np.dtype, pd.Series, pd.api.extensions.ExtensionDtype
    ],
    make_series: bool = True,
):
    if dtypes is None:
        return None
    elif isinstance(dtypes, np.dtype):
        return dtypes
    elif isinstance(dtypes, list):
        val = [make_dtype(dt) for dt in dtypes]
        return val if not make_series else pd.Series(val)
    elif isinstance(dtypes, dict):
        val = {k: make_dtype(v) for k, v in dtypes.items()}
        return val if not make_series else pd.Series(val)
    elif isinstance(dtypes, pd.Series):
        return dtypes.map(make_dtype)
    else:
        return make_dtype(dtypes)


def serialize_serializable(serializable, compress: bool = False):
    from .serialization import serialize

    bio = io.BytesIO()
    header, buffers = serialize(serializable)
    buf_sizes = [getattr(buf, "nbytes", len(buf)) for buf in buffers]
    header[0]["buf_sizes"] = buf_sizes

    def encode_np_num(obj):
        if isinstance(obj, np.generic) and obj.shape == () and not np.isnan(obj):
            return obj.item()
        return obj

    s_header = msgpack.dumps(header, default=encode_np_num)

    bio.write(struct.pack("<Q", len(s_header)))
    bio.write(s_header)
    for buf in buffers:
        bio.write(buf)
    ser_graph = bio.getvalue()

    if compress:
        ser_graph = zlib.compress(ser_graph)
    return ser_graph


def deserialize_serializable(ser_serializable: bytes):
    from .serialization import deserialize

    bio = io.BytesIO(ser_serializable)
    s_header_length = struct.unpack("Q", bio.read(8))[0]
    header2 = msgpack.loads(bio.read(s_header_length))
    buffers2 = [bio.read(s) for s in header2[0]["buf_sizes"]]
    return deserialize(header2, buffers2)


def skip_na_call(func: Callable):
    @functools.wraps(func)
    def new_func(x):
        return func(x) if x is not None else None

    return new_func


def url_path_join(*pieces):
    """Join components of url into a relative url

    Use to prevent double slash when joining subpath. This will leave the
    initial and final / in place
    """
    initial = pieces[0].startswith("/")
    final = pieces[-1].endswith("/")
    stripped = [s.strip("/") for s in pieces]
    result = "/".join(s for s in stripped if s)
    if initial:
        result = "/" + result
    if final:
        result = result + "/"
    if result == "//":
        result = "/"
    return result


def random_ports(port: int, n: int):
    """Generate a list of n random ports near the given port.

    The first 5 ports will be sequential, and the remaining n-5 will be
    randomly selected in the range [port-2*n, port+2*n].
    """
    for i in range(min(5, n)):
        yield port + i
    for i in range(n - 5):
        yield max(1, port + random.randint(-2 * n, 2 * n))


def build_temp_table_name(session_id: str, tileable_key: str) -> str:
    return f"tmp_mf_{session_id}_{tileable_key}"


def build_temp_intermediate_table_name(session_id: str, tileable_key: str) -> str:
    temp_table = build_temp_table_name(session_id, tileable_key)
    return f"{temp_table}_intermediate"


def build_session_volume_name(session_id: str) -> str:
    return f"mf_vol_{session_id.replace('-', '_')}"


async def wait_http_response(
    url: str, *, request_timeout: TimeoutType = None, **kwargs
) -> httpclient.HTTPResponse:
    start_time = time.time()
    while request_timeout is None or time.time() - start_time < request_timeout:
        timeout_left = min(10.0, time.time() - start_time) if request_timeout else None
        try:
            return await httpclient.AsyncHTTPClient().fetch(
                url, request_timeout=timeout_left, **kwargs
            )
        except HTTPTimeoutError:
            pass
    raise TimeoutError


def get_handler_timeout_value(handler: web.RequestHandler) -> TimeoutType:
    wait = bool(int(handler.get_argument("wait", "0")))
    timeout = float(handler.get_argument("timeout", "0"))
    if wait and abs(timeout) < 1e-6:
        timeout = None
    elif not wait:
        timeout = 0
    return timeout


def format_timeout_params(timeout: TimeoutType) -> str:
    if timeout is None:
        return "?wait=1"
    elif abs(timeout) < 1e-6:
        return "?wait=0"
    else:
        return f"?wait=1&timeout={timeout}"


def unwrap_partial_function(func):
    while isinstance(func, functools.partial):
        func = func.func
    return func


_PrimitiveType = TypeVar("_PrimitiveType")


def create_sync_primitive(
    cls: Type[_PrimitiveType], loop: asyncio.AbstractEventLoop
) -> _PrimitiveType:
    """
    Create an asyncio sync primitive (locks, events, etc.)
    in a certain event loop.
    """
    if sys.version_info[1] < 10:
        return cls(loop=loop)

    # From Python3.10 the loop parameter has been removed. We should work around here.
    try:
        old_loop = asyncio.get_event_loop()
    except RuntimeError:
        old_loop = None
    try:
        asyncio.set_event_loop(loop)
        primitive = cls()
    finally:
        asyncio.set_event_loop(old_loop)
    return primitive


class ToThreadCancelledError(asyncio.CancelledError):
    def __init__(self, *args, result=None):
        super().__init__(*args)
        self._result = result

    @property
    def result(self):
        return self._result


_ToThreadRetType = TypeVar("_ToThreadRetType")


class ToThreadMixin:
    _thread_pool_size = 1
    _counter = itertools.count().__next__

    def __del__(self):
        if hasattr(self, "_pool"):
            kw = {"wait": False}
            if sys.version_info[:2] >= (3, 9):
                kw["cancel_futures"] = True
            self._pool.shutdown(**kw)

    async def to_thread(
        self,
        func: Callable[..., _ToThreadRetType],
        *args,
        wait_on_cancel: bool = False,
        timeout: float = None,
        debug_task_name: Optional[str] = None,
        **kwargs,
    ) -> _ToThreadRetType:
        if not hasattr(self, "_pool"):
            self._pool = concurrent.futures.ThreadPoolExecutor(
                self._thread_pool_size,
                thread_name_prefix=f"{type(self).__name__}Pool-{self._counter()}",
            )

        loop = asyncio.events.get_running_loop()
        ctx = contextvars.copy_context()
        func_call = functools.partial(ctx.run, func, *args, **kwargs)
        fut = loop.run_in_executor(self._pool, func_call)

        if loop.get_debug():
            # create a task and mark its name
            default_task_name = None
            try:
                unwrapped = unwrap_partial_function(func)
                default_task_name = unwrapped.__qualname__
                if getattr(unwrapped, "__module__", None):
                    default_task_name = unwrapped.__module__ + "#" + default_task_name
            except:  # noqa # pragma: no cover
                try:
                    default_task_name = repr(func)
                except:  # noqa
                    pass
            debug_task_name = debug_task_name or default_task_name

            async def _wait_fut(aio_fut):
                return await aio_fut

            fut = asyncio.create_task(_wait_fut(fut))
            if sys.version_info[:2] == (3, 7):
                # In Python3.7 we should hack the task name to print it in debug logs.
                setattr(fut, "fd_task_name", debug_task_name)
            else:
                fut.set_name(debug_task_name)

        try:
            coro = fut
            if wait_on_cancel:
                coro = asyncio.shield(coro)
            if timeout is not None:
                coro = asyncio.wait_for(coro, timeout)
            return await coro
        except (asyncio.CancelledError, asyncio.TimeoutError) as ex:
            if not wait_on_cancel:
                raise
            result = await fut
            raise ToThreadCancelledError(*ex.args, result=result)

    def ensure_async_call(
        self,
        func: Callable[..., _ToThreadRetType],
        *args,
        wait_on_cancel: bool = False,
        **kwargs,
    ) -> Awaitable[_ToThreadRetType]:
        if asyncio.iscoroutinefunction(func):
            return func(*args, **kwargs)
        return self.to_thread(func, *args, wait_on_cancel=wait_on_cancel, **kwargs)


class PatchableMixin:
    """Patch not None field to dest_obj"""

    __slots__ = ()

    _patchable_attrs = tuple()

    def patch_to(self, dest_obj) -> None:
        for attr in self._patchable_attrs:
            val = getattr(self, attr, None)
            if val is not None:
                setattr(dest_obj, attr, val)


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


def to_hashable(obj: Any) -> Hashable:
    if isinstance(obj, Mapping):
        items = type(obj)((k, to_hashable(v)) for k, v in obj.items())
    elif not isinstance(obj, str) and isinstance(obj, Iterable):
        items = tuple(to_hashable(item) for item in obj)
    elif isinstance(obj, Hashable):
        items = obj
    else:
        raise TypeError(type(obj))
    return items


def estimate_pandas_size(
    pd_obj, max_samples: int = 10, min_sample_rows: int = 100
) -> int:
    if len(pd_obj) <= min_sample_rows or isinstance(pd_obj, pd.RangeIndex):
        return sys.getsizeof(pd_obj)
    if isinstance(pd_obj, pd.MultiIndex):
        # MultiIndex's sample size can't be used to estimate
        return sys.getsizeof(pd_obj)

    def _is_fast_dtype(dtype):
        if isinstance(dtype, np.dtype):
            return np.issubdtype(dtype, np.number)
        else:
            return isinstance(dtype, ArrowDtype)

    dtypes = []
    is_series = False
    if isinstance(pd_obj, pd.DataFrame):
        dtypes.extend(pd_obj.dtypes)
        index_obj = pd_obj.index
    elif isinstance(pd_obj, pd.Series):
        dtypes.append(pd_obj.dtype)
        index_obj = pd_obj.index
        is_series = True
    else:
        index_obj = pd_obj

    # handling possible MultiIndex
    if hasattr(index_obj, "dtypes"):
        dtypes.extend(index_obj.dtypes)
    else:
        dtypes.append(index_obj.dtype)

    if all(_is_fast_dtype(dtype) for dtype in dtypes):
        return sys.getsizeof(pd_obj)

    indices = np.sort(np.random.choice(len(pd_obj), size=max_samples, replace=False))
    iloc = pd_obj if isinstance(pd_obj, pd.Index) else pd_obj.iloc
    if isinstance(index_obj, pd.MultiIndex):
        # MultiIndex's sample size is much greater than expected, thus we calculate
        # the size separately.
        index_size = sys.getsizeof(pd_obj.index)
        if is_series:
            sample_frame_size = iloc[indices].memory_usage(deep=True, index=False)
        else:
            sample_frame_size = iloc[indices].memory_usage(deep=True, index=False).sum()
        return index_size + sample_frame_size * len(pd_obj) // max_samples
    else:
        sample_size = sys.getsizeof(iloc[indices])
        return sample_size * len(pd_obj) // max_samples


def estimate_table_size(odps_entry, full_table_name: str, partitions: List[str] = None):
    try:
        data_src = odps_entry.get_table(full_table_name)
        if isinstance(partitions, str):
            partitions = [partitions]
        if not partitions:
            size_mul = 1
        else:
            size_mul = len(partitions)
            data_src = data_src.partitions[partitions[0]]
        return size_mul * data_src.size
    except:
        return float("inf")


class ModulePlaceholder:
    def __init__(self, mod_name: str):
        self._mod_name = mod_name

    def _raises(self):
        raise AttributeError(f"{self._mod_name} is required but not installed.")

    def __getattr__(self, key):
        self._raises()

    def __call__(self, *_args, **_kwargs):
        self._raises()


def lazy_import(
    name: str,
    package: str = None,
    globals: Dict = None,  # pylint: disable=redefined-builtin
    locals: Dict = None,  # pylint: disable=redefined-builtin
    rename: str = None,
    placeholder: bool = False,
):
    rename = rename or name
    prefix_name = name.split(".", 1)[0]
    globals = globals or inspect.currentframe().f_back.f_globals

    class LazyModule(object):
        def __init__(self):
            self._on_loads = []

        def __getattr__(self, item):
            if item.startswith("_pytest") or item in ("__bases__", "__test__"):
                raise AttributeError(item)

            real_mod = importlib.import_module(name, package=package)
            if rename in globals:
                globals[rename] = real_mod
            elif locals is not None:
                locals[rename] = real_mod
            ret = getattr(real_mod, item)
            for on_load_func in self._on_loads:
                on_load_func()
            # make sure on_load hooks only executed once
            self._on_loads = []
            return ret

        def add_load_handler(self, func: Callable):
            self._on_loads.append(func)
            return func

    if pkgutil.find_loader(prefix_name) is not None:
        return LazyModule()
    elif placeholder:
        return ModulePlaceholder(prefix_name)
    else:
        return None


def sbytes(x: Any) -> bytes:
    # NB: bytes() in Python 3 has different semantic with Python 2, see: help(bytes)
    from numbers import Number

    if x is None or isinstance(x, Number):
        return bytes(str(x), encoding="ascii")
    elif isinstance(x, list):
        return bytes("[" + ", ".join([str(k) for k in x]) + "]", encoding="utf-8")
    elif isinstance(x, tuple):
        return bytes("(" + ", ".join([str(k) for k in x]) + ")", encoding="utf-8")
    elif isinstance(x, str):
        return bytes(x, encoding="utf-8")
    else:
        try:
            return bytes(x)
        except TypeError:
            return bytes(str(x), encoding="utf-8")


def is_full_slice(slc: Any) -> bool:
    """Check if the input is a full slice ((:) or (0:))"""
    return (
        isinstance(slc, slice)
        and (slc.start == 0 or slc.start is None)
        and slc.stop is None
        and slc.step is None
    )


_enter_counter = 0
_initial_session = None


def enter_current_session(func: Callable):
    @functools.wraps(func)
    def wrapped(cls, ctx, op):
        from .session import AbstractSession, get_default_session

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


# from https://github.com/ericvsmith/dataclasses/blob/master/dataclass_tools.py
# released under Apache License 2.0
def dataslots(cls):
    # Need to create a new class, since we can't set __slots__
    #  after a class has been created.

    # Make sure __slots__ isn't already set.
    if "__slots__" in cls.__dict__:  # pragma: no cover
        raise TypeError(f"{cls.__name__} already specifies __slots__")

    # Create a new dict for our new class.
    cls_dict = dict(cls.__dict__)
    field_names = tuple(f.name for f in dataclasses.fields(cls))
    cls_dict["__slots__"] = field_names
    for field_name in field_names:
        # Remove our attributes, if present. They'll still be
        #  available in _MARKER.
        cls_dict.pop(field_name, None)
    # Remove __dict__ itself.
    cls_dict.pop("__dict__", None)
    # And finally create the class.
    qualname = getattr(cls, "__qualname__", None)
    cls = type(cls)(cls.__name__, cls.__bases__, cls_dict)
    if qualname is not None:
        cls.__qualname__ = qualname
    return cls


def adapt_docstring(doc: str) -> str:
    """
    Adapt numpy-style docstrings to MaxFrame docstring.

    This util function will add MaxFrame imports, replace object references
    and add execute calls. Note that check is needed after replacement.
    """
    if doc is None:
        return None

    lines = []
    first_prompt = True
    prev_prompt = False
    has_numpy = "np." in doc
    has_pandas = "pd." in doc

    for line in doc.splitlines():
        sp = line.strip()
        if sp.startswith(">>>") or sp.startswith("..."):
            prev_prompt = True
            if first_prompt:
                first_prompt = False
                indent = "".join(itertools.takewhile(lambda x: x in (" ", "\t"), line))
                if has_numpy:
                    lines.extend([indent + ">>> import maxframe.tensor as mt"])
                if has_pandas:
                    lines.extend([indent + ">>> import maxframe.dataframe as md"])
            line = line.replace("np.", "mt.").replace("pd.", "md.")
        elif prev_prompt:
            prev_prompt = False
            if sp and lines[-1].strip().strip("."):
                # need prev line contains chars other than dots
                lines[-1] += ".execute()"
        lines.append(line)
    return "\n".join(lines)


def stringify_path(path: Union[str, os.PathLike]) -> str:
    """
    Convert *path* to a string or unicode path if possible.
    """
    if isinstance(path, str):
        return path

    # checking whether path implements the filesystem protocol
    try:
        return path.__fspath__()
    except AttributeError:
        raise TypeError("not a path-like object")


_memory_size_indices = {"": 0, "b": 0, "k": 1, "m": 2, "g": 3, "t": 4}

_size_pattern = re.compile(r"^([0-9.-]+)\s*([a-z]*)$")


def parse_readable_size(value: Union[str, int, float]) -> Tuple[float, bool]:
    """
    Parse a human-readable size representation into a numeric value.

    This function converts various size formats into their corresponding
    float values. It supports:
    - Raw numbers (e.g., 1024)
    - Percentages (e.g., "50%")
    - Size units (e.g., "10KB", "5.5GB", "2MiB")

    The function recognizes standard binary prefixes (K, M, G, T, etc.) and
    handles different suffix variations (B, iB, etc.).

    Parameters
    ----------
    value : Union[str, int, float]
        The size value to parse, can be a string, int, or float

    Returns
    -------
    Tuple[float, bool]
        A tuple of (parsed_value, is_percentage)
        - parsed_value: The parsed numeric value
        - is_percentage: True if the input was a percentage, False otherwise
    """
    if isinstance(value, numbers.Number):
        return float(value), False

    if not isinstance(value, str):
        raise TypeError(f"Expected string or number, got {type(value).__name__}")

    value = value.strip().lower()

    # Handle percentage values
    if value.endswith("%"):
        return float(value[:-1]) / 100, True

    # Parse the value into numeric and unit parts
    match = _size_pattern.match(value)
    if not match:
        raise ValueError(f"Cannot parse size value: {value}")

    number_str, unit = match.groups()

    # convert to float
    number = float(number_str)

    # if no unit, return the number
    if not unit:
        return number, False

    # Validate the unit prefix
    if unit[0] not in _memory_size_indices:
        valid_prefixes = ", ".join(sorted(_memory_size_indices.keys()))
        raise ValueError(
            f"Unknown unit prefix: '{unit[0]}', valid prefixes are {valid_prefixes}"
        )

    # Check for valid unit suffix
    if len(unit) > 1 and unit[1:] not in ("ib", "b", "i", ""):
        raise ValueError(f"Invalid size unit suffix: {unit}")

    is_binary_unit = "i" in unit.lower()
    # calc the multiplier
    base = 1024 if is_binary_unit else 1000
    multiplier = base ** _memory_size_indices[unit[0]]

    return number * multiplier, False


def parse_size_to_megabytes(
    value: Union[str, int, float], default_number_unit: str = "GiB"
) -> float:
    try:
        value = float(value)
    except BaseException:
        pass

    if isinstance(value, numbers.Number):
        if not default_number_unit:
            raise ValueError(
                "`default_number_unit` must be provided when give a number value"
            )
        return parse_size_to_megabytes(
            f"{value}{default_number_unit}", default_number_unit
        )

    bytes_number, is_percentage = parse_readable_size(value)
    if is_percentage:
        raise ValueError("Percentage size is not supported to parse")

    return bytes_number / (1024**2)  # convert to megabytes


def remove_suffix(value: str, suffix: str) -> Tuple[str, bool]:
    """
    Remove a suffix from a given string if it exists.

    Parameters
    ----------
    value : str
        The original string.
    suffix : str
        The suffix to be removed.

    Returns
    -------
    Tuple[str, bool]
        A tuple containing the modified string and a boolean indicating whether the suffix was found.
    """

    # Check if the suffix is an empty string
    if len(suffix) == 0:
        # If the suffix is empty, return the original string with True
        return value, True

    # Check if the length of the value is less than the length of the suffix
    if len(value) < len(suffix):
        # If the value is shorter than the suffix, it cannot have the suffix
        return value, False

    # Check if the suffix matches the end of the value
    match = value.endswith(suffix)

    # If the suffix is found, remove it; otherwise, return the original string
    if match:
        return value[: -len(suffix)], match
    else:
        return value, match


def find_objects(
    nested: Union[List, Dict],
    types: Union[None, Type, Tuple[Type]] = None,
    checker: Callable[..., bool] = None,
) -> List:
    found = []
    stack = [nested]

    while len(stack) > 0:
        it = stack.pop()
        if (types is not None and isinstance(it, types)) or (
            checker is not None and checker(it)
        ):
            found.append(it)
            continue

        if isinstance(it, (list, tuple, set)):
            stack.extend(list(it)[::-1])
        elif isinstance(it, dict):
            stack.extend(list(it.values())[::-1])

    return found


def replace_objects(nested: Union[List, Dict], mapping: Mapping) -> Union[List, Dict]:
    if not mapping:
        return nested

    if isinstance(nested, dict):
        vals = list(nested.values())
    else:
        vals = list(nested)

    new_vals = []
    for val in vals:
        if isinstance(val, (dict, list, tuple, set)):
            new_val = replace_objects(val, mapping)
        else:
            try:
                new_val = mapping.get(val, val)
            except TypeError:
                new_val = val
        new_vals.append(new_val)

    if isinstance(nested, dict):
        return type(nested)((k, v) for k, v in zip(nested.keys(), new_vals))
    else:
        return type(nested)(new_vals)


def trait_from_env(
    trait_name: str, env: str, trait: Optional[traitlets.TraitType] = None
):
    if trait is None:
        prev_locals = inspect.stack()[1].frame.f_locals
        trait = prev_locals[trait_name]

    default_value = trait.default_value
    sub_trait: traitlets.TraitType = getattr(trait, "_trait", None)

    def default_value_simple(self):
        env_val = os.getenv(env, default_value)
        if isinstance(env_val, (str, bytes)):
            return trait.from_string(env_val)
        return env_val

    def default_value_list(self):
        env_val = os.getenv(env, default_value)
        if env_val is None or isinstance(env_val, traitlets.Sentinel):
            return env_val

        parts = env_val.split(",") if env_val else []
        if sub_trait:
            return [sub_trait.from_string(s) for s in parts]
        else:
            return parts

    if isinstance(trait, traitlets.List):
        default_value_fun = default_value_list
    else:  # pragma: no cover
        default_value_fun = default_value_simple

    default_value_fun.__name__ = trait_name + "_default"
    return traitlets.default(trait_name)(default_value_fun)


def relay_future(
    dest: Union[asyncio.Future, concurrent.futures.Future],
    src: Union[asyncio.Future, concurrent.futures.Future],
) -> None:
    def cb(fut: Union[asyncio.Future, concurrent.futures.Future]):
        try:
            dest.set_result(fut.result())
        except BaseException as ex:
            dest.set_exception(ex)

    src.add_done_callback(cb)


_arrow_type_constructors = {}
if pa:
    _arrow_type_constructors = {
        "bool": pa.bool_,
        "list": lambda x: pa.list_(dict(x)["item"]),
        "map": lambda x: pa.map_(*x),
        "struct": pa.struct,
        "fixed_size_binary": pa.binary,
        "halffloat": pa.float16,
        "float": pa.float32,
        "double": pa.float64,
        "decimal": pa.decimal128,
        # repr() of date32 and date64 has `day` or `ms`
        #  which is not needed in constructors
        "date32": lambda *_: pa.date32(),
        "date64": lambda *_: pa.date64(),
    }
    _plain_arrow_types = """
    null
    int8 int16 int32 int64
    uint8 uint16 uint32 uint64
    float16 float32 float64
    decimal128 decimal256
    string utf8 binary
    time32 time64 duration timestamp
    month_day_nano_interval
    """
    for _type_name in _plain_arrow_types.split():
        try:
            _arrow_type_constructors[_type_name] = getattr(pa, _type_name)
        except AttributeError:  # pragma: no cover
            pass


def arrow_type_from_str(type_str: str) -> pa.DataType:
    """
    Convert arrow type representations (for inst., list<item: int64>)
    into arrow DataType instances
    """
    # enable consecutive brackets to be tokenized
    type_str = type_str.replace("<", "< ").replace(">", " >")
    token_iter = pytokenize.tokenize(io.BytesIO(type_str.encode()).readline)
    value_stack, op_stack = [], []

    def _pop_make_type(with_args: bool = False, combined: bool = True):
        """
        Pops tops of value stacks, creates a DataType instance and push back

        Parameters
        ----------
            with_args: bool
                if True, will contain next item (parameter list) in
                the value stack as parameters
            combined: bool
                if True, will use first element of the top of the value stack
                in DataType constructors
        """
        args = () if not with_args else (value_stack.pop(-1),)
        if not combined:
            args = args[0]
        type_name = value_stack.pop(-1)
        if isinstance(type_name, pa.DataType):
            value_stack.append(type_name)
        elif type_name in _arrow_type_constructors:
            value_stack.append(_arrow_type_constructors[type_name](*args))
        else:  # pragma: no cover
            value_stack.append(type_name)

    def _pop_make_struct_field():
        """parameterized sub-types need to be represented as tuples"""
        nonlocal value_stack

        op_stack.pop(-1)
        if isinstance(value_stack[-1], str) and value_stack[-1].lower() in (
            "null",
            "not null",
        ):
            values = value_stack[-3:]
            value_stack = value_stack[:-3]
            values[-1] = values[-1] == "null"
        else:
            values = value_stack[-2:]
            value_stack = value_stack[:-2]
        value_stack.append(tuple(values))

    for token in token_iter:
        if token.type == pytokenize.OP:
            if token.string == ":":
                op_stack.append(token.string)
            elif token.string == ",":
                # gather previous sub-types
                if op_stack[-1] in ("<", ":"):
                    _pop_make_type()
                if op_stack[-1] == ":":
                    _pop_make_struct_field()

                # put generated item into the parameter list
                val = value_stack.pop(-1)
                value_stack[-1].append(val)
            elif token.string in ("<", "[", "("):
                # pushes an empty parameter list for future use
                value_stack.append([])
                op_stack.append(token.string)
            elif token.string in (")", "]"):
                # put generated item into the parameter list
                val = value_stack.pop(-1)
                value_stack[-1].append(val)
                # make DataType (i.e., fixed_size_binary / decimal) given args
                _pop_make_type(with_args=True, combined=False)
                op_stack.pop(-1)
            elif token.string == ">":
                _pop_make_type()
                if op_stack[-1] == ":":
                    _pop_make_struct_field()

                # put generated item into the parameter list
                val = value_stack.pop(-1)
                value_stack[-1].append(val)
                # make DataType (i.e., list / map / struct) given args
                _pop_make_type(with_args=True)
                op_stack.pop(-1)
        elif token.type == pytokenize.NAME:
            if value_stack and value_stack[-1] == "not":
                value_stack[-1] += " " + token.string
            else:
                value_stack.append(token.string)
        elif token.type == pytokenize.NUMBER:
            value_stack.append(int(token.string))
        elif token.type == pytokenize.ENDMARKER:
            # make final type
            _pop_make_type()
    if len(value_stack) > 1:
        raise ValueError(f"Cannot parse type {type_str}")
    return value_stack[-1]


def get_python_tag():
    # todo add implementation suffix for non-GIL tags when PEP703 is ready
    version_info = sys.version_info
    return f"cp{version_info[0]}{version_info[1]}"


def get_item_if_scalar(val: Any) -> Any:
    if isinstance(val, np.ndarray) and val.shape == ():
        return val.item()
    return val


def collect_leaf_operators(root) -> List[Type]:
    result = []

    def _collect(op_type):
        if len(op_type.__subclasses__()) == 0:
            result.append(op_type)
        for subclass in op_type.__subclasses__():
            _collect(subclass)

    _collect(root)
    return result


@contextmanager
def sync_pyodps_options():
    from odps.config import option_context as pyodps_option_context

    from .config import options

    with pyodps_option_context() as cfg:
        cfg.local_timezone = options.local_timezone
        if options.session.enable_schema:
            cfg.enable_schema = options.session.enable_schema
        yield


def str_to_bool(s: Optional[str]) -> Optional[bool]:
    return s.lower().strip() in ("true", "1") if isinstance(s, str) else s


def is_empty(val):
    if isinstance(val, (pd.DataFrame, pd.Series, pd.Index)):
        return val.empty
    return not bool(val)


def extract_class_name(cls):
    return cls.__module__ + "#" + cls.__qualname__


def flatten(nested_iterable: Union[List, Tuple]) -> List:
    """
    Flatten a nested iterable into a list.

    Parameters
    ----------
    nested_iterable : list or tuple
        an iterable which can contain other iterables

    Returns
    -------
    flattened : list

    Examples
    --------
    >>> flatten([[0, 1], [2, 3]])
    [0, 1, 2, 3]
    >>> flatten([[0, 1], [[3], [4, 5]]])
    [0, 1, 3, 4, 5]
    """

    flattened = []
    stack = list(nested_iterable)[::-1]
    while len(stack) > 0:
        inp = stack.pop()
        if isinstance(inp, (tuple, list)):
            stack.extend(inp[::-1])
        else:
            flattened.append(inp)
    return flattened


def stack_back(flattened: List, raw: Union[List, Tuple]) -> Union[List, Tuple]:
    """
    Organize a new iterable from a flattened list according to raw iterable.

    Parameters
    ----------
    flattened : list
        flattened list
    raw: list
        raw iterable

    Returns
    -------
    ret : list

    Examples
    --------
    >>> raw = [[0, 1], [2, [3, 4]]]
    >>> flattened = flatten(raw)
    >>> flattened
    [0, 1, 2, 3, 4]
    >>> a = [f + 1 for f in flattened]
    >>> a
    [1, 2, 3, 4, 5]
    >>> stack_back(a, raw)
    [[1, 2], [3, [4, 5]]]
    """
    flattened_iter = iter(flattened)
    result = list()

    def _stack(container, items):
        for item in items:
            if not isinstance(item, (list, tuple)):
                container.append(next(flattened_iter))
            else:
                new_container = list()
                container.append(new_container)
                _stack(new_container, item)

        return container

    return _stack(result, raw)


_RetryRetType = TypeVar("_RetryRetType")


def call_with_retry(
    func: Callable[..., _RetryRetType],
    *args,
    retry_times: Optional[int] = None,
    retry_timeout: TimeoutType = None,
    delay: TimeoutType = None,
    reset_func: Optional[Callable] = None,
    exc_type: Union[
        Type[BaseException], Tuple[Type[BaseException], ...]
    ] = BaseException,
    allow_interrupt: bool = True,
    no_raise: bool = False,
    is_func_async: Optional[bool] = None,
    **kwargs,
) -> _RetryRetType:
    """
    Retry calling function given specified times or timeout.

    Parameters
    ----------
    func: Callable
        function to be retried
    args
        arguments to be passed to the function
    retry_times: Optional[int]
        times to retry the function
    retry_timeout: TimeoutType
        timeout in seconds to retry the function
    delay: TimeoutType
        delay in seconds between every trial
    reset_func: Callable
        Function to call after every trial
    exc_type: Type[BaseException] | Tuple[Type[BaseException], ...]
        Exception type for retrial
    allow_interrupt: bool
        If True, KeyboardInterrupt will stop the retry
    no_raise: bool
        If True, no exception will be raised even if all trials failed
    is_func_async: bool
        If True, func will be treated as async
    kwargs
        keyword arguments to be passed to the function

    Returns
    -------
    Return value of the original function
    """
    from .config import options

    retry_num = 0
    retry_times = retry_times if retry_times is not None else options.retry_times
    delay = delay if delay is not None else options.retry_delay
    start_time = time.monotonic() if retry_timeout is not None else None

    def raise_or_continue(exc: BaseException):
        nonlocal retry_num
        retry_num += 1
        if allow_interrupt and isinstance(exc, KeyboardInterrupt):
            raise exc from None
        if (retry_times is not None and retry_num > retry_times) or (
            retry_timeout is not None
            and start_time is not None
            and time.monotonic() - start_time > retry_timeout
        ):
            if no_raise:
                return sys.exc_info()
            raise exc from None

    async def async_retry():
        while True:
            try:
                return await func(*args, **kwargs)
            except exc_type as ex:
                await asyncio.sleep(delay)
                res = raise_or_continue(ex)
                if res is not None:
                    return res

                if callable(reset_func):
                    reset_res = reset_func()
                    if asyncio.iscoroutine(reset_res):
                        await reset_res

    def sync_retry():
        while True:
            try:
                return func(*args, **kwargs)
            except exc_type as ex:
                time.sleep(delay)
                res = raise_or_continue(ex)
                if res is not None:
                    return res
                if callable(reset_func):
                    reset_func()

    unwrap_func = func
    if is_func_async is None:
        # unwrap to get true result if func is async
        while isinstance(unwrap_func, functools.partial):
            unwrap_func = unwrap_func.func

    if is_func_async or asyncio.iscoroutinefunction(unwrap_func):
        return async_retry()
    else:
        return sync_retry()


def update_wlm_quota_settings(session_id: str, engine_settings: Dict[str, Any]):
    from .config import options

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


def copy_if_possible(obj: Any, deep=False) -> Any:
    try:
        return copy.deepcopy(obj) if deep else copy.copy(obj)
    except:  # pragma: no cover
        return obj


def cache_tileables(*tileables):
    from .core import ENTITY_TYPE

    if len(tileables) == 1 and isinstance(tileables[0], (tuple, list)):
        tileables = tileables[0]
    for t in tileables:
        if isinstance(t, ENTITY_TYPE):
            t.cache = True


def ignore_warning(func: Callable):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return inner


class ServiceLoggerAdapter(logging.LoggerAdapter):
    extra_key_mapping = {}

    def process(self, msg, kwargs):
        merged_extra = (self.extra or {}).copy()
        merged_extra.update(kwargs)

        prefix = " ".join(
            f"{self.extra_key_mapping.get(k) or k.capitalize()}={merged_extra[k]}"
            for k in merged_extra.keys()
        )
        msg = f"[{prefix}] {msg}"
        return msg, kwargs


@contextmanager
def atomic_writer(filename, mode="w", **kwargs):
    """
    Write to a file in an atomic way.
    """
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filename) or ".")
    os.chmod(temp_path, 0o644)
    os.close(temp_fd)  # Close the file descriptor immediately and we reopen this later.

    try:
        # Write to temp file.
        with open(temp_path, mode, **kwargs) as temp_file:
            yield temp_file

        # Replace the original file with the temp file atomically.
        os.replace(temp_path, filename)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def prevent_called_from_pandas(level=2):
    """Prevent method from being called from pandas"""
    frame = sys._getframe(level)
    called_frame = sys._getframe(1)
    pd_pack_location = os.path.dirname(pd.__file__)
    if frame.f_code.co_filename.startswith(pd_pack_location):
        raise AttributeError(called_frame.f_code.co_name)


def combine_error_message_and_traceback(
    messages: List[str], tracebacks: List[List[str]]
) -> str:
    tbs = []
    for msg, tb in zip(messages, tracebacks):
        tbs.append("".join([msg + "\n"] + tb))
    return "\nCaused by:\n".join(tbs)


def generate_unique_id(byte_len: int) -> Generator[str, None, None]:
    """
    The ids are ensured to be unique in one generator.
    DO NOT use this generator in global scope or singleton class members,
    as it may not free the set.
    """
    generated_ids = set()
    while True:
        new_id = new_random_id(byte_len).hex()
        if new_id not in generated_ids:
            generated_ids.add(new_id)
            yield new_id


def validate_and_adjust_resource_ratio(
    expect_resources: Dict[str, Any],
    max_memory_cpu_ratio: float = None,
    adjust: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    """
    Validate and optionally adjust CPU:memory ratio to meet maximum requirements.

    Args:
        expect_resources: Dictionary containing resource specifications
        max_memory_cpu_ratio: Maximum memory/cpu ratio (if None, will use config value)
        adjust: Whether to automatically adjust resources to meet ratio

    Returns:
        Tuple of (adjusted_resources, was_adjusted)
    """
    cpu = expect_resources.get("cpu") or 1
    memory = expect_resources.get("memory")

    if cpu is None or memory is None or max_memory_cpu_ratio is None:
        return expect_resources, False

    # Convert memory to GiB if it's a string
    cpu = max(cpu, 1)
    memory_gib = parse_size_to_megabytes(memory, default_number_unit="GiB") / 1024
    current_ratio = memory_gib / cpu

    if current_ratio > max_memory_cpu_ratio:
        # Adjust CPU to meet maximum ratio, don't reduce resources
        recommended_cpu = math.ceil(memory_gib / max_memory_cpu_ratio)
        new_ratio = memory_gib / recommended_cpu
        if adjust:
            adjusted_resources = expect_resources.copy()
            adjusted_resources["cpu"] = recommended_cpu

            warnings.warn(
                f"UDF resource auto-adjustment: Current UDF settings"
                f" (CPU: {cpu}, Memory: {memory_gib}Gib, Ratio: {current_ratio:.2f})"
                f" exceed maximum allowed ratio {max_memory_cpu_ratio:.1f}. "
                f"Automatically adjusted to (CPU: {recommended_cpu},"
                f" Memory: {memory_gib:.2f}:1Gib,"
                f" Ratio: {new_ratio:.2f}:1) to meet requirements."
            )
            return adjusted_resources, True
        else:
            warnings.warn(
                f"UDF resource ratio warning: Current UDF settings"
                f" (CPU: {cpu}, Memory: {memory_gib}Gib, Ratio: {current_ratio:.2f})"
                f" exceed maximum allowed ratio {max_memory_cpu_ratio:.1f}. "
                f"Consider adjusting CPU to at least {recommended_cpu}"
                f" (which would result in Ratio: {new_ratio:.2f}) to meet requirements."
            )

    return expect_resources, False


def get_pd_option(option_name, default=no_default):
    """Get pandas option. If not exist return `default`."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return pd.get_option(option_name)
    except (KeyError, AttributeError):
        if default is no_default:
            raise
        return default


@contextlib.contextmanager
def pd_option_context(*args):
    arg_kv = dict(zip(args[0::2], args[1::2]))
    new_args = []
    for k, v in arg_kv.items():
        try:
            get_pd_option(k)
        except (KeyError, AttributeError):  # pragma: no cover
            continue
        new_args.extend([k, v])
    if not new_args:  # pragma: no cover
        yield
    else:
        with pd.option_context(*new_args):
            yield
