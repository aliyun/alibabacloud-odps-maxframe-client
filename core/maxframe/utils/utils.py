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
import copy
import dataclasses
import enum
import importlib
import inspect
import itertools
import logging
import math
import numbers
import os
import pkgutil
import random
import re
import sys
import tempfile
import warnings
from collections.abc import Hashable, Mapping
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd
import traitlets
from tornado import web

from ..lib.dtypes_extension import ArrowDtype
from ..lib.version import parse as parse_version
from ..typing_ import TileableType, TimeoutType
from ._utils_c import new_random_id

_is_ci = (os.environ.get("CI") or "0").lower() in ("1", "true")
np_release_version: Tuple[int] = parse_version(np.__version__).release
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


class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


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
    check_unexpected_kwargs(kwargs)

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


class PatchableMixin:
    """Patch not None field to dest_obj"""

    __slots__ = ()

    _patchable_attrs = tuple()

    def patch_to(self, dest_obj) -> None:
        for attr in self._patchable_attrs:
            val = getattr(self, attr, None)
            if val is not None:
                setattr(dest_obj, attr, val)


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
            return isinstance(dtype, ArrowDtype) or dtype == pd.StringDtype("pyarrow")

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


def str_to_bool(s: Optional[str]) -> Optional[bool]:
    return s.lower().strip() in ("true", "1") if isinstance(s, str) else s


def extract_class_name(cls):
    return cls.__module__ + "#" + cls.__qualname__


def copy_if_possible(obj: Any, deep=False) -> Any:
    try:
        return copy.deepcopy(obj) if deep else copy.copy(obj)
    except:  # pragma: no cover
        return obj


def cache_tileables(*tileables):
    from ..core import ENTITY_TYPE

    if len(tileables) == 1 and isinstance(tileables[0], (tuple, list)):
        tileables = tileables[0]
    for t in tileables:
        if isinstance(t, ENTITY_TYPE):
            t.cache = True


class ServiceLoggerAdapter(logging.LoggerAdapter):
    extra_key_mapping = {}

    def process(self, msg, kwargs):
        merged_extra = (self.extra or {}).copy()
        merged_extra.update(kwargs)
        merged_extra.pop("exc_info", None)

        for k in self.extra_key_mapping:
            kwargs.pop(k, None)

        prefix = " ".join(
            f"{self.extra_key_mapping.get(k) or k.capitalize()}={merged_extra[k]}"
            for k in merged_extra.keys()
        )
        msg = f"[{prefix}] {msg}"
        return msg, kwargs


@contextmanager
def atomic_writer(filename, mode="w", permissions: int = 0o644, **kwargs):
    """
    Write to a file in an atomic way.
    """
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filename) or ".")
    os.chmod(temp_path, permissions)
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


def check_unexpected_kwargs(kwargs):
    if kwargs:
        caller_func = sys._getframe(1).f_code.co_name
        raise TypeError(
            f"{caller_func}() got an unexpected keyword argument '{next(iter(kwargs))}'"
        )


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


class KeyLogWrapper:
    def __init__(self, chunks, limit=None):
        self._chunks = chunks
        self._limit = limit

    def __str__(self):
        chunks = disp_chunks = list(self._chunks)
        if self._limit:
            disp_chunks = self._chunks[: self._limit]
        strs = ", ".join(
            f"{c.key[:8] if not isinstance(c, str) else c[:8]}..." for c in disp_chunks
        )
        if self._limit and len(chunks) > self._limit:
            strs = f"{strs}, and {len(chunks) - self._limit} more..."
        return f"[{strs}]"
