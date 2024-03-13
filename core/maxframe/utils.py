# Copyright 1999-2024 Alibaba Group Holding Ltd.
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
import contextvars
import dataclasses
import datetime
import enum
import functools
import hashlib
import importlib
import inspect
import io
import itertools
import numbers
import os
import pkgutil
import random
import struct
import sys
import threading
import time
import tokenize as pytokenize
import traceback
import types
import weakref
import zlib
from collections.abc import Hashable, Mapping
from contextlib import contextmanager
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
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
from .lib.version import parse as parse_version
from .typing_ import ChunkType, EntityType, TileableType, TimeoutType

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
    if shape:
        return tuple(s if not np.isnan(s) else -1 for s in shape)
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


def has_unknown_shape(*tiled_tileables: TileableType) -> bool:
    for tileable in tiled_tileables:
        if getattr(tileable, "shape", None) is None:
            continue
        if any(pd.isnull(s) for s in tileable.shape):
            return True
        if any(pd.isnull(s) for s in itertools.chain(*tileable.nsplits)):
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


def build_fetch_chunk(chunk: ChunkType, **kwargs) -> ChunkType:
    from .core.operator import ShuffleProxy

    chunk_op = chunk.op
    params = chunk.params.copy()
    assert not isinstance(chunk_op, ShuffleProxy)
    # for non-shuffle nodes, we build Fetch chunks
    # to replace original chunk
    op = chunk_op.get_fetch_op_cls(chunk)(sparse=chunk.op.sparse, gpu=chunk.op.gpu)
    return op.new_chunk(
        None,
        is_broadcaster=chunk.is_broadcaster,
        kws=[params],
        _key=chunk.key,
        **kwargs,
    )


def build_fetch_tileable(tileable: TileableType) -> TileableType:
    if tileable.is_coarse():
        chunks = None
    else:
        chunks = []
        for c in tileable.chunks:
            fetch_chunk = build_fetch_chunk(c, index=c.index)
            chunks.append(fetch_chunk)

    tileable_op = tileable.op
    params = tileable.params.copy()

    new_op = tileable_op.get_fetch_op_cls(tileable)(_id=tileable_op.id)
    return new_op.new_tileables(
        None,
        chunks=chunks,
        nsplits=tileable.nsplits,
        _key=tileable.key,
        _id=tileable.id,
        **params,
    )[0]


def build_fetch(entity: EntityType) -> EntityType:
    from .core import CHUNK_TYPE, ENTITY_TYPE

    if isinstance(entity, CHUNK_TYPE):
        return build_fetch_chunk(entity)
    elif isinstance(entity, ENTITY_TYPE):
        return build_fetch_tileable(entity)
    else:
        raise TypeError(f"Type {type(entity)} not supported")


def get_dtype(dtype: Union[np.dtype, pd.api.extensions.ExtensionDtype]):
    if pd.api.types.is_extension_array_dtype(dtype):
        return dtype
    elif dtype is pd.Timestamp or dtype is datetime.datetime:
        return np.dtype("datetime64[ns]")
    elif dtype is pd.Timedelta or dtype is datetime.timedelta:
        return np.dtype("timedelta64[ns]")
    else:
        return np.dtype(dtype)


def serialize_serializable(serializable, compress: bool = False):
    from .serialization import serialize

    bio = io.BytesIO()
    header, buffers = serialize(serializable)
    buf_sizes = [getattr(buf, "nbytes", len(buf)) for buf in buffers]
    header[0]["buf_sizes"] = buf_sizes
    s_header = msgpack.dumps(header)
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


def build_session_volume_name(session_id: str) -> str:
    return f"mf_vol_{session_id}"


def build_tileable_dir_name(tileable_key: str) -> str:
    m = hashlib.md5()
    m.update(f"mf_dir_{tileable_key}".encode())
    return m.hexdigest()


def extract_messages_and_stacks(exc: Exception) -> Tuple[List[str], List[str]]:
    cur_exc = exc
    messages, stacks = [], []
    while True:
        messages.append(str(cur_exc))
        stacks.append("".join(traceback.format_tb(cur_exc.__traceback__)))
        if exc.__cause__ is None:
            break
        cur_exc = exc.__cause__
    return messages, stacks


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


async def to_thread_pool(func, *args, pool=None, **kwargs):
    loop = asyncio.events.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(pool, func_call)


class ToThreadCancelledError(asyncio.CancelledError):
    def __init__(self, *args, result=None):
        super().__init__(*args)
        self._result = result

    @property
    def result(self):
        return self._result


_ToThreadRetType = TypeVar("_ToThreadRetType")


class ToThreadMixin:
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
        **kwargs,
    ) -> _ToThreadRetType:
        if not hasattr(self, "_pool"):
            self._pool = concurrent.futures.ThreadPoolExecutor(1)

        task = asyncio.create_task(
            to_thread_pool(func, *args, **kwargs, pool=self._pool)
        )
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError) as ex:
            if not wait_on_cancel:
                raise
            result = await task
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


def config_odps_default_options():
    from odps import options as odps_options

    odps_options.sql.settings = {
        "odps.longtime.instance": "false",
        "odps.sql.session.select.only": "false",
        "metaservice.client.cache.enable": "false",
        "odps.sql.session.result.cache.enable": "false",
        "odps.sql.submit.mode": "script",
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

    from .dataframe.arrays import ArrowDtype

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
        return bytes(x)


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
            cvars = tuple([x.cell_contents for x in func.__closure__])
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
            if sp:
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


_memory_size_indices = {"": 0, "k": 1, "m": 2, "g": 3, "t": 4}


def parse_readable_size(value: Union[str, int, float]) -> Tuple[float, bool]:
    if isinstance(value, numbers.Number):
        return float(value), False

    value = value.strip().lower()
    num_pos = 0
    while num_pos < len(value) and value[num_pos] in "0123456789.-":
        num_pos += 1

    value, suffix = value[:num_pos], value[num_pos:]
    suffix = suffix.strip()
    if suffix.endswith("%"):
        return float(value) / 100, True

    try:
        return float(value) * (1024 ** _memory_size_indices[suffix[:1]]), False
    except (ValueError, KeyError):
        raise ValueError(f"Unknown limitation value: {value}")


def remove_suffix(value: str, suffix: str) -> str:
    return value[: -len(suffix)] if value.endswith(suffix) else value


def find_objects(nested: Union[List, Dict], types: Union[Type, Tuple[Type]]) -> List:
    found = []
    stack = [nested]

    while len(stack) > 0:
        it = stack.pop()
        if isinstance(it, types):
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
    }
    _plain_arrow_types = """
    null
    int8 int16 int32 int64
    uint8 uint16 uint32 uint64
    float16 float32 float64
    date32 date64
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

    def _pop_make_type(with_args: bool = False, combined: bool = True) -> None:
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

    for token in token_iter:
        if token.type == pytokenize.OP:
            if token.string == ":":
                op_stack.append(token.string)
            elif token.string == ",":
                # gather previous sub-types
                if op_stack[-1] in ("<", ":"):
                    _pop_make_type()

                if op_stack[-1] == ":":
                    # parameterized sub-types need to be represented as tuples
                    op_stack.pop(-1)
                    values = value_stack[-2:]
                    value_stack = value_stack[:-2]
                    value_stack.append(tuple(values))
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
                    # parameterized sub-types need to be represented as tuples
                    op_stack.pop(-1)
                    values = value_stack[-2:]
                    value_stack = value_stack[:-2]
                    value_stack.append(tuple(values))

                # put generated item into the parameter list
                val = value_stack.pop(-1)
                value_stack[-1].append(val)
                # make DataType (i.e., list / map / struct) given args
                _pop_make_type(True)
                op_stack.pop(-1)
        elif token.type == pytokenize.NAME:
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
