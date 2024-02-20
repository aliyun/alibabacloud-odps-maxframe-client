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

import functools
import inspect
import itertools
import logging
import operator
import sys
from contextlib import contextmanager
from numbers import Integral
from typing import Any, Callable, List

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype
from pandas.api.types import is_string_dtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.inference import is_dict_like, is_list_like

from ..core import Entity, ExecutableTuple
from ..lib.mmh3 import hash as mmh_hash
from ..utils import (
    ModulePlaceholder,
    is_full_slice,
    lazy_import,
    parse_version,
    sbytes,
    tokenize,
)

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = ModulePlaceholder("pyarrow")

cudf = lazy_import("cudf", rename="cudf")
vineyard = lazy_import("vineyard")
try:
    import ray

    ray_release_version = parse_version(ray.__version__).release
    ray_deprecate_ml_dataset = ray_release_version[:2] >= (2, 0)
except ImportError:
    ray_release_version = None
    ray_deprecate_ml_dataset = None
logger = logging.getLogger(__name__)


def hash_index(index, size):
    def func(x, size):
        return mmh_hash(sbytes(x)) % size

    f = functools.partial(func, size=size)
    idx_to_grouped = index.groupby(index.map(f))
    return [idx_to_grouped.get(i, list()) for i in range(size)]


def hash_dataframe_on(df, on, size, level=None):
    if on is None:
        idx = df.index
        if level is not None:
            idx = idx.to_frame(False)[level]
        if cudf and isinstance(idx, cudf.Index):  # pragma: no cover
            idx = idx.to_pandas()
        hashed_label = pd.util.hash_pandas_object(idx, categorize=False)
    elif callable(on):
        # todo optimization can be added, if ``on`` is a numpy ufunc or sth can be vectorized
        hashed_label = pd.util.hash_pandas_object(df.index.map(on), categorize=False)
    else:
        if isinstance(on, list):
            to_concat = []
            for v in on:
                if isinstance(v, pd.Series):
                    to_concat.append(v)
                else:
                    to_concat.append(df[v])
            data = pd.concat(to_concat, axis=1)
        else:
            data = df[on]
        hashed_label = pd.util.hash_pandas_object(data, index=False, categorize=False)
    idx_to_grouped = pd.RangeIndex(0, len(hashed_label)).groupby(hashed_label % size)
    return [idx_to_grouped.get(i, pd.Index([])) for i in range(size)]


def hash_dtypes(dtypes, size):
    hashed_indexes = hash_index(dtypes.index, size)
    return [dtypes[index] for index in hashed_indexes]


def sort_dataframe_inplace(df, *axis):
    for ax in axis:
        df.sort_index(axis=ax, inplace=True)
    return df


@functools.lru_cache(1)
def _get_range_index_type():
    if cudf is not None:
        return pd.RangeIndex, cudf.RangeIndex
    else:
        return pd.RangeIndex


@functools.lru_cache(1)
def _get_multi_index_type():
    if cudf is not None:
        return pd.MultiIndex, cudf.MultiIndex
    else:
        return pd.MultiIndex


def _get_range_index_start(pd_range_index):
    try:
        return pd_range_index.start
    except AttributeError:  # pragma: no cover
        return pd_range_index._start


def _get_range_index_stop(pd_range_index):
    try:
        return pd_range_index.stop
    except AttributeError:  # pragma: no cover
        return pd_range_index._stop


def _get_range_index_step(pd_range_index):
    try:
        return pd_range_index.step
    except AttributeError:  # pragma: no cover
        pass
    try:  # pragma: no cover
        return pd_range_index._step
    except AttributeError:  # pragma: no cover
        return 1  # cudf does not support step arg


def is_pd_range_empty(pd_range_index):
    start, stop, step = (
        _get_range_index_start(pd_range_index),
        _get_range_index_stop(pd_range_index),
        _get_range_index_step(pd_range_index),
    )
    return (start >= stop and step >= 0) or (start <= stop and step < 0)


def parse_index(index_value, *args, store_data=False, key=None):
    from .core import IndexValue

    def _extract_property(index, tp, ret_data):
        kw = {
            "_min_val": _get_index_min(index),
            "_max_val": _get_index_max(index),
            "_min_val_close": True,
            "_max_val_close": True,
            "_key": key or _tokenize_index(index, *args),
        }
        if ret_data:
            kw["_data"] = index.values
        for field in tp._FIELDS:
            if field in kw or field == "_data":
                continue
            val = getattr(index, field.lstrip("_"), None)
            if val is not None:
                kw[field] = val
        return kw

    def _tokenize_index(index, *token_objects):
        if not index.empty:
            return tokenize(index)
        else:
            return tokenize(index, *token_objects)

    def _get_index_min(index):
        try:
            return index.min()
        except (ValueError, AttributeError):
            if isinstance(index, pd.IntervalIndex):
                return None
            raise
        except TypeError:
            return None

    def _get_index_max(index):
        try:
            return index.max()
        except (ValueError, AttributeError):
            if isinstance(index, pd.IntervalIndex):
                return None
            raise
        except TypeError:
            return None

    def _serialize_index(index):
        tp = getattr(IndexValue, type(index).__name__)
        properties = _extract_property(index, tp, store_data)
        properties["_name"] = index.name
        return tp(**properties)

    def _serialize_range_index(index):
        if is_pd_range_empty(index):
            properties = {
                "_is_monotonic_increasing": True,
                "_is_monotonic_decreasing": False,
                "_is_unique": True,
                "_min_val": _get_index_min(index),
                "_max_val": _get_index_max(index),
                "_min_val_close": True,
                "_max_val_close": False,
                "_key": key or _tokenize_index(index, *args),
                "_name": index.name,
                "_dtype": index.dtype,
            }
        else:
            properties = _extract_property(index, IndexValue.RangeIndex, False)
        return IndexValue.RangeIndex(
            _slice=slice(
                _get_range_index_start(index),
                _get_range_index_stop(index),
                _get_range_index_step(index),
            ),
            **properties,
        )

    def _serialize_multi_index(index):
        kw = _extract_property(index, IndexValue.MultiIndex, store_data)
        kw["_sortorder"] = index.sortorder
        kw["_dtypes"] = [lev.dtype for lev in index.levels]
        return IndexValue.MultiIndex(**kw)

    if index_value is None:
        return IndexValue(
            _index_value=IndexValue.Index(
                _is_monotonic_increasing=False,
                _is_monotonic_decreasing=False,
                _is_unique=False,
                _min_val=None,
                _max_val=None,
                _min_val_close=True,
                _max_val_close=True,
                _key=key or tokenize(*args),
            )
        )
    if hasattr(index_value, "to_pandas"):  # pragma: no cover
        # convert cudf.Index to pandas
        index_value = index_value.to_pandas()

    if isinstance(index_value, _get_range_index_type()):
        return IndexValue(_index_value=_serialize_range_index(index_value))
    elif isinstance(index_value, _get_multi_index_type()):
        return IndexValue(_index_value=_serialize_multi_index(index_value))
    else:
        return IndexValue(_index_value=_serialize_index(index_value))


def gen_unknown_index_value(index_value, *args):
    pd_index = index_value.to_pandas()
    if isinstance(pd_index, pd.RangeIndex):
        return parse_index(pd.RangeIndex(-1), *args)
    elif not isinstance(pd_index, pd.MultiIndex):
        return parse_index(pd.Index([], dtype=pd_index.dtype), *args)
    else:
        i = pd.MultiIndex.from_arrays(
            [c[:0] for c in pd_index.levels], names=pd_index.names
        )
        return parse_index(i, *args)


def split_monotonic_index_min_max(
    left_min_max, left_increase, right_min_max, right_increase
):
    """
    Split the original two min_max into new min_max. Each min_max should be a list
    in which each item should be a 4-tuple indicates that this chunk's min value,
    whether the min value is close, the max value, and whether the max value is close.
    The return value would be a nested list, each item is a list
    indicates that how this chunk should be split into.

    :param left_min_max: the left min_max
    :param left_increase: if the original data of left is increased
    :param right_min_max: the right min_max
    :param right_increase: if the original data of right is increased
    :return: nested list in which each item indicates how min_max is split

    >>> left_min_max = [(0, True, 3, True), (4, True, 8, True), (12, True, 18, True),
    ...                 (20, True, 22, True)]
    >>> right_min_max = [(2, True, 6, True), (7, True, 9, True), (10, True, 14, True),
    ...                  (18, True, 19, True)]
    >>> l, r = split_monotonic_index_min_max(left_min_max, True, right_min_max, True)
    >>> l
    [[(0, True, 2, False), (2, True, 3, True)], [(3, False, 4, False), (4, True, 6, True), (6, False, 7, False),
    (7, True, 8, True)], [(8, False, 9, True), (10, True, 12, False), (12, True, 14, True), (14, False, 18, False),
    (18, True, 18, True)], [(18, False, 19, True), [20, True, 22, True]]]
    >>> r
    [[(0, True, 2, False), (2, True, 3, True), (3, False, 4, False), (4, True, 6, True)],
    [(6, False, 7, False), (7, True, 8, True), (8, False, 9, True)], [(10, True, 12, False), (12, True, 14, True)],
    [(14, False, 18, False), (18, True, 18, True), (18, False, 19, True), [20, True, 22, True]]]
    """
    left_idx_to_min_max = [[] for _ in left_min_max]
    right_idx_to_min_max = [[] for _ in right_min_max]
    left_curr_min_max = list(left_min_max[0])
    right_curr_min_max = list(right_min_max[0])
    left_curr_idx = right_curr_idx = 0
    left_terminate = right_terminate = False

    while not left_terminate or not right_terminate:
        if left_terminate:
            left_idx_to_min_max[left_curr_idx].append(tuple(right_curr_min_max))
            right_idx_to_min_max[right_curr_idx].append(tuple(right_curr_min_max))
            if right_curr_idx + 1 >= len(right_min_max):
                right_terminate = True
            else:
                right_curr_idx += 1
                right_curr_min_max = list(right_min_max[right_curr_idx])
        elif right_terminate:
            right_idx_to_min_max[right_curr_idx].append(tuple(left_curr_min_max))
            left_idx_to_min_max[left_curr_idx].append(tuple(left_curr_min_max))
            if left_curr_idx + 1 >= len(left_min_max):
                left_terminate = True
            else:
                left_curr_idx += 1
                left_curr_min_max = list(left_min_max[left_curr_idx])
        elif left_curr_min_max[0] < right_curr_min_max[0]:
            # left min < right min
            right_min = [right_curr_min_max[0], not right_curr_min_max[1]]
            max_val = min(left_curr_min_max[2:], right_min)
            assert len(max_val) == 2
            min_max = (
                left_curr_min_max[0],
                left_curr_min_max[1],
                max_val[0],
                max_val[1],
            )
            left_idx_to_min_max[left_curr_idx].append(min_max)
            right_idx_to_min_max[right_curr_idx].append(min_max)
            if left_curr_min_max[2:] == max_val:
                # left max < right min
                if left_curr_idx + 1 >= len(left_min_max):
                    left_terminate = True
                else:
                    left_curr_idx += 1
                    left_curr_min_max = list(left_min_max[left_curr_idx])
            else:
                # from left min(left min close) to right min(exclude right min close)
                left_curr_min_max[:2] = right_curr_min_max[:2]
        elif left_curr_min_max[0] > right_curr_min_max[0]:
            # left min > right min
            left_min = [left_curr_min_max[0], not left_curr_min_max[1]]
            max_val = min(right_curr_min_max[2:], left_min)
            min_max = (
                right_curr_min_max[0],
                right_curr_min_max[1],
                max_val[0],
                max_val[1],
            )
            left_idx_to_min_max[left_curr_idx].append(min_max)
            right_idx_to_min_max[right_curr_idx].append(min_max)
            if right_curr_min_max[2:] == max_val:
                # right max < left min
                if right_curr_idx + 1 >= len(right_min_max):
                    right_terminate = True
                else:
                    right_curr_idx += 1
                    right_curr_min_max = list(right_min_max[right_curr_idx])
            else:
                # from left min(left min close) to right min(exclude right min close)
                right_curr_min_max[:2] = left_curr_min_max[:2]
        else:
            # left min == right min
            max_val = min(left_curr_min_max[2:], right_curr_min_max[2:])
            assert len(max_val) == 2
            min_max = (
                left_curr_min_max[0],
                left_curr_min_max[1],
                max_val[0],
                max_val[1],
            )
            left_idx_to_min_max[left_curr_idx].append(min_max)
            right_idx_to_min_max[right_curr_idx].append(min_max)
            if max_val == left_curr_min_max[2:]:
                if left_curr_idx + 1 >= len(left_min_max):
                    left_terminate = True
                else:
                    left_curr_idx += 1
                    left_curr_min_max = list(left_min_max[left_curr_idx])
            else:
                left_curr_min_max[:2] = max_val[0], not max_val[1]
            if max_val == right_curr_min_max[2:]:
                if right_curr_idx + 1 >= len(right_min_max):
                    right_terminate = True
                else:
                    right_curr_idx += 1
                    right_curr_min_max = list(right_min_max[right_curr_idx])
            else:
                right_curr_min_max[:2] = max_val[0], not max_val[1]

    if left_increase is False:
        left_idx_to_min_max = list(reversed(left_idx_to_min_max))
    if right_increase is False:
        right_idx_to_min_max = list(reversed(right_idx_to_min_max))

    return left_idx_to_min_max, right_idx_to_min_max


def build_split_idx_to_origin_idx(splits, increase=True):
    # splits' len is equal to the original chunk size on a specified axis,
    # splits is sth like [[(0, True, 2, True), (2, False, 3, True)]]
    # which means there is one input chunk, and will be split into 2 out chunks
    # in this function, we want to build a new dict from the out chunk index to
    # the original chunk index and the inner position, like {0: (0, 0), 1: (0, 1)}
    if increase is False:
        splits = list(reversed(splits))
    out_idx = itertools.count(0)
    res = dict()
    for origin_idx, _ in enumerate(splits):
        for pos in range(len(splits[origin_idx])):
            if increase is False:
                o_idx = len(splits) - origin_idx - 1
            else:
                o_idx = origin_idx
            res[next(out_idx)] = o_idx, pos
    return res


def _generate_value(dtype, fill_value):
    # special handle for datetime64 and timedelta64
    dispatch = {
        np.datetime64: pd.Timestamp,
        np.timedelta64: pd.Timedelta,
        pd.CategoricalDtype.type: lambda x: pd.CategoricalDtype([x]),
        # for object, we do not know the actual dtype,
        # just convert to str for common usage
        np.object_: lambda x: str(fill_value),
    }
    # otherwise, just use dtype.type itself to convert
    convert = dispatch.get(dtype.type, dtype.type)
    return convert(fill_value)


def build_empty_df(dtypes, index=None):
    columns = dtypes.index
    length = len(index) if index is not None else 0
    record = [[_generate_value(dtype, 1) for dtype in dtypes]] * max(1, length)

    # duplicate column may exist,
    # so use RangeIndex first
    df = pd.DataFrame(record, columns=range(len(dtypes)), index=index)
    for i, dtype in enumerate(dtypes):
        s = df.iloc[:, i]
        if not pd.api.types.is_dtype_equal(s.dtype, dtype):
            df.iloc[:, i] = s.astype(dtype)

    df.columns = columns
    return df[:length] if len(df) > length else df


def build_df(df_obj, fill_value=1, size=1, ensure_string=False):
    dfs = []
    if not isinstance(size, (list, tuple)):
        sizes = [size]
    else:
        sizes = size

    if not isinstance(fill_value, (list, tuple)):
        fill_values = [fill_value]
    else:
        fill_values = fill_value

    from .core import SERIES_TYPE

    dtypes = (
        pd.Series([df_obj.dtype], index=[df_obj.name])
        if isinstance(df_obj, SERIES_TYPE)
        else df_obj.dtypes
    )
    for size, fill_value in zip(sizes, fill_values):
        record = [[_generate_value(dtype, fill_value) for dtype in dtypes]] * size
        df = pd.DataFrame(record)
        df.columns = dtypes.index

        if len(record) != 0:  # columns is empty in some cases
            target_index = df_obj.index_value.to_pandas()
            if isinstance(target_index, pd.MultiIndex):
                index_val = tuple(
                    _generate_value(level.dtype, fill_value)
                    for level in target_index.levels
                )
                df.index = pd.MultiIndex.from_tuples(
                    [index_val] * size, names=target_index.names
                )
            else:
                index_val = _generate_value(target_index.dtype, fill_value)
                df.index = pd.Index([index_val] * size, name=target_index.name)

        # make sure dtypes correct
        for i, dtype in enumerate(dtypes):
            s = df.iloc[:, i]
            if not pd.api.types.is_dtype_equal(s.dtype, dtype):
                df[df.columns[i]] = s.astype(dtype)
        dfs.append(df)
    if len(dfs) == 1:
        ret_df = dfs[0]
    else:
        ret_df = pd.concat(dfs)

    if ensure_string:
        obj_dtypes = dtypes[dtypes == np.dtype("O")]
        ret_df[obj_dtypes.index] = ret_df[obj_dtypes.index].radd("O")
    return ret_df


def build_empty_series(dtype, index=None, name=None):
    length = len(index) if index is not None else 0
    return pd.Series(
        [_generate_value(dtype, 1) for _ in range(length)],
        dtype=dtype,
        index=index,
        name=name,
    )


def build_series(
    series_obj=None,
    fill_value=1,
    size=1,
    name=None,
    ensure_string=False,
    dtype=None,
    index=None,
):
    seriess = []
    if not isinstance(size, (list, tuple)):
        sizes = [size]
    else:
        sizes = size

    if not isinstance(fill_value, (list, tuple)):
        fill_values = [fill_value]
    else:
        fill_values = fill_value

    if series_obj is not None:
        dtype = series_obj.dtype
        try:
            series_index = series_obj.index_value.to_pandas()[:0]
        except AttributeError:
            series_index = series_obj.index[:0]
    else:
        series_index = index[:0] if index is not None else None

    for size, fill_value in zip(sizes, fill_values):
        empty_series = build_empty_series(dtype, name=name, index=series_index)
        record = _generate_value(dtype, fill_value)
        if isinstance(empty_series.index, pd.MultiIndex):
            index = tuple(
                _generate_value(level.dtype, fill_value)
                for level in empty_series.index.levels
            )
            empty_series = empty_series.reindex(
                index=pd.MultiIndex.from_tuples([index], names=empty_series.index.names)
            )
            empty_series.iloc[0] = record
        else:
            if isinstance(empty_series.index.dtype, pd.CategoricalDtype):
                index = None
            else:
                index = _generate_value(empty_series.index.dtype, fill_value)
            empty_series.loc[index] = record

        empty_series = pd.concat([empty_series] * size)
        # make sure dtype correct for MultiIndex
        empty_series = empty_series.astype(dtype, copy=False)
        seriess.append(empty_series)

    if len(seriess) == 1:
        ret_series = seriess[0]
    else:
        ret_series = pd.concat(seriess)

    if ensure_string and dtype == np.dtype("O"):
        ret_series = ret_series.radd("O")
    return ret_series


def infer_index_value(left_index_value, right_index_value):
    from .core import IndexValue

    if isinstance(left_index_value.value, IndexValue.RangeIndex) and isinstance(
        right_index_value.value, IndexValue.RangeIndex
    ):
        if left_index_value.value.slice == right_index_value.value.slice:
            return left_index_value
        return parse_index(
            pd.Index([], dtype=np.int64), left_index_value, right_index_value
        )

    # when left index and right index is identical, and both of them are elements unique,
    # we can infer that the out index should be identical also
    if (
        left_index_value.is_unique
        and right_index_value.is_unique
        and left_index_value.key == right_index_value.key
    ):
        return left_index_value

    left_index = left_index_value.to_pandas()
    right_index = right_index_value.to_pandas()
    out_index = pd.Index(
        [], dtype=find_common_type([left_index.dtype, right_index.dtype])
    )
    return parse_index(out_index, left_index_value, right_index_value)


def indexing_index_value(index_value, indexes, store_data=False, rechunk=False):
    pd_index = index_value.to_pandas()
    # when rechunk is True, the output index shall be treated
    # different from the input one
    if not rechunk and isinstance(indexes, slice) and is_full_slice(indexes):
        return index_value
    elif not index_value.has_value():
        new_index_value = parse_index(pd_index, indexes, store_data=store_data)
        new_index_value._index_value._min_val = index_value.min_val
        new_index_value._index_value._min_val_close = index_value.min_val_close
        new_index_value._index_value._max_val = index_value.max_val
        new_index_value._index_value._max_val_close = index_value.max_val_close
        return new_index_value
    else:
        if isinstance(indexes, Integral):
            return parse_index(pd_index[[indexes]], store_data=store_data)
        elif isinstance(indexes, Entity):
            if isinstance(pd_index, pd.RangeIndex):
                return parse_index(
                    pd.RangeIndex(-1), indexes, index_value, store_data=False
                )
            else:
                return parse_index(
                    type(pd_index)([]), indexes, index_value, store_data=False
                )
        if isinstance(indexes, tuple):
            return parse_index(pd_index[list(indexes)], store_data=store_data)
        else:
            return parse_index(pd_index[indexes], store_data=store_data)


def merge_index_value(to_merge_index_values: dict, store_data: bool = False):
    """
    Merge index value according to their chunk index.

    Parameters
    ----------
    to_merge_index_values : dict
        index to index_value
    store_data : bool
        store data in index_value

    Returns
    -------
    merged_index_value
    """

    pd_index = None
    min_val, min_val_close, max_val, max_val_close = None, None, None, None
    for _, chunk_index_value in sorted(to_merge_index_values.items()):
        if pd_index is None:
            pd_index = chunk_index_value.to_pandas()
            min_val, min_val_close, max_val, max_val_close = (
                chunk_index_value.min_val,
                chunk_index_value.min_val_close,
                chunk_index_value.max_val,
                chunk_index_value.max_val_close,
            )
        else:
            cur_pd_index = chunk_index_value.to_pandas()
            if store_data or (
                isinstance(pd_index, pd.RangeIndex)
                and isinstance(cur_pd_index, pd.RangeIndex)
                and cur_pd_index.step == pd_index.step
                and cur_pd_index.start == pd_index.stop
            ):
                # range index that is continuous
                pd_index = pd_index.append(cur_pd_index)
            else:
                pd_index = pd.Index([], dtype=pd_index.dtype)
            if chunk_index_value.min_val is not None:
                try:
                    if min_val is None or min_val > chunk_index_value.min_val:
                        min_val = chunk_index_value.min_val
                        min_val_close = chunk_index_value.min_val_close
                except TypeError:
                    # min_value has different types that cannot compare
                    # just stop compare
                    continue
            if chunk_index_value.max_val is not None:
                if max_val is None or max_val < chunk_index_value.max_val:
                    max_val = chunk_index_value.max_val
                    max_val_close = chunk_index_value.max_val_close

    index_value = parse_index(pd_index, store_data=store_data)
    if not index_value.has_value():
        index_value._index_value._min_val = min_val
        index_value._index_value._min_val_close = min_val_close
        index_value._index_value._max_val = max_val
        index_value._index_value._max_val_close = max_val_close
    return index_value


def infer_dtypes(left_dtypes, right_dtypes, operator):
    left = build_empty_df(left_dtypes)
    right = build_empty_df(right_dtypes)
    return operator(left, right).dtypes


@functools.lru_cache(100)
def infer_dtype(left_dtype, right_dtype, operator):
    left = build_empty_series(left_dtype)
    right = build_empty_series(right_dtype)
    return operator(left, right).dtype


def filter_dtypes(dtypes, column_min_max):
    left_filter = operator.ge if column_min_max[1] else operator.gt
    left = left_filter(dtypes.index, column_min_max[0])
    right_filter = operator.le if column_min_max[3] else operator.lt
    right = right_filter(dtypes.index, column_min_max[2])
    return dtypes[left & right]


def in_range_index(i, pd_range_index):
    """
    Check whether the input `i` is within `pd_range_index` which is a pd.RangeIndex.
    """
    start, stop, step = (
        _get_range_index_start(pd_range_index),
        _get_range_index_stop(pd_range_index),
        _get_range_index_step(pd_range_index),
    )
    if step > 0 and start <= i < stop and (i - start) % step == 0:
        return True
    if step < 0 and start >= i > stop and (start - i) % step == 0:
        return True
    return False


def wrap_notimplemented_exception(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotImplementedError:
            return NotImplemented

    return wrapper


def validate_axis(axis, tileable=None):
    if axis == "index":
        axis = 0
    elif axis == "columns":
        axis = 1

    illegal = False
    try:
        axis = operator.index(axis)
        if axis < 0 or (tileable is not None and axis >= tileable.ndim):
            illegal = True
    except TypeError:
        illegal = True

    if illegal:
        raise ValueError(f"No axis named {axis} for object type {type(tileable)}")
    return axis


def validate_axis_style_args(
    data, args, kwargs, arg_name, method_name
):  # pragma: no cover
    """Argument handler for mixed index, columns / axis functions

    In an attempt to handle both `.method(index, columns)`, and
    `.method(arg, axis=.)`, we have to do some bad things to argument
    parsing. This translates all arguments to `{index=., columns=.}` style.

    Parameters
    ----------
    data : DataFrame
    args : tuple
        All positional arguments from the user
    kwargs : dict
        All keyword arguments from the user
    arg_name, method_name : str
        Used for better error messages

    Returns
    -------
    kwargs : dict
        A dictionary of keyword arguments. Doesn't modify ``kwargs``
        inplace, so update them with the return value here.
    """
    out = {}
    # Goal: fill 'out' with index/columns-style arguments
    # like out = {'index': foo, 'columns': bar}

    # Start by validating for consistency
    axes_names = ["index"] if data.ndim == 1 else ["index", "columns"]
    if "axis" in kwargs and any(x in kwargs for x in axes_names):
        msg = "Cannot specify both 'axis' and any of 'index' or 'columns'."
        raise TypeError(msg)

    # First fill with explicit values provided by the user...
    if arg_name in kwargs:
        if args:
            msg = f"{method_name} got multiple values for argument '{arg_name}'"
            raise TypeError(msg)

        axis = axes_names[validate_axis(kwargs.get("axis", 0), data)]
        out[axis] = kwargs[arg_name]

    # More user-provided arguments, now from kwargs
    for k, v in kwargs.items():
        try:
            ax = axes_names[validate_axis(k, data)]
        except ValueError:
            pass
        else:
            out[ax] = v

    # All user-provided kwargs have been handled now.
    # Now we supplement with positional arguments, emitting warnings
    # when there's ambiguity and raising when there's conflicts

    if len(args) == 0:
        pass  # It's up to the function to decide if this is valid
    elif len(args) == 1:
        axis = axes_names[validate_axis(kwargs.get("axis", 0), data)]
        out[axis] = args[0]
    elif len(args) == 2:
        if "axis" in kwargs:
            # Unambiguously wrong
            msg = "Cannot specify both 'axis' and any of 'index' or 'columns'"
            raise TypeError(msg)

        msg = (
            "Interpreting call\n\t'.{method_name}(a, b)' as "
            "\n\t'.{method_name}(index=a, columns=b)'.\nUse named "
            "arguments to remove any ambiguity."
        )
        raise TypeError(msg.format(method_name=method_name))
    else:
        msg = f"Cannot specify all of '{arg_name}', 'index', 'columns'."
        raise TypeError(msg)
    return out


def validate_output_types(**kwargs):
    from ..core import OutputType

    output_type = kwargs.pop("object_type", None) or kwargs.pop("output_type", None)
    output_types = kwargs.pop("output_types", None) or (
        [output_type] if output_type is not None else None
    )
    return (
        [
            getattr(OutputType, v.lower()) if isinstance(v, str) else v
            for v in output_types
        ]
        if output_types
        else None
    )


def fetch_corner_data(df_or_series, session=None) -> pd.DataFrame:
    """
    Fetch corner DataFrame or Series for repr usage.

    :param df_or_series: DataFrame or Series
    :return: corner DataFrame
    """
    from .indexing.iloc import iloc

    max_rows = pd.get_option("display.max_rows")
    try:
        min_rows = pd.get_option("display.min_rows")
        min_rows = min(min_rows, max_rows)
    except KeyError:  # pragma: no cover
        # display.min_rows is introduced in pandas 0.25
        min_rows = max_rows

    index_size = None
    if (
        df_or_series.shape[0] > max_rows
        and df_or_series.shape[0] > min_rows // 2 * 2 + 2
    ):
        # for pandas, greater than max_rows
        # will display min_rows
        # thus we fetch min_rows + 2 lines
        index_size = min_rows // 2 + 1

    if index_size is None:
        return df_or_series._fetch(session=session)
    else:
        head = iloc(df_or_series)[:index_size]
        tail = iloc(df_or_series)[-index_size:]
        head_data, tail_data = ExecutableTuple([head, tail]).fetch(session=session)
        xdf = cudf if head.op.is_gpu() else pd
        return xdf.concat([head_data, tail_data], axis="index")


class ReprSeries(pd.Series):
    def __init__(self, corner_data, real_shape):
        super().__init__(corner_data)
        self._real_shape = real_shape

    def __len__(self):
        # As we only fetch corner data to repr,
        # the length would be wrong and we have no way to control,
        # thus we just overwrite the length to show the real one
        return self._real_shape[0]


def filter_dtypes_by_index(dtypes, index):
    try:
        new_dtypes = dtypes.loc[index].dropna()
    except KeyError:
        dtypes_idx = (
            dtypes.index.to_frame()
            .merge(index.to_frame())
            .set_index(list(range(dtypes.index.nlevels)))
            .index
        )
        new_dtypes = dtypes.loc[dtypes_idx]
        new_dtypes.index.names = dtypes.index.names
    return new_dtypes


@contextmanager
def create_sa_connection(con, **kwargs):
    import sqlalchemy as sa
    from sqlalchemy.engine import Connection, Engine

    # process con
    engine = None
    if isinstance(con, Connection):
        # connection create by user
        close = False
        dispose = False
    elif isinstance(con, Engine):
        con = con.connect()
        close = True
        dispose = False
    else:
        engine = sa.create_engine(con, **kwargs)
        con = engine.connect()
        close = True
        dispose = True

    try:
        yield con
    finally:
        if close:
            con.close()
        if dispose:
            engine.dispose()


def to_arrow_dtypes(dtypes, test_df=None):
    from .arrays import ArrowStringDtype

    new_dtypes = dtypes.copy()
    for i in range(len(dtypes)):
        dtype = dtypes.iloc[i]
        if is_string_dtype(dtype):
            if test_df is not None:
                series = test_df.iloc[:, i]
                # check value
                non_na_series = series[series.notna()]
                if len(non_na_series) > 0:
                    first_value = non_na_series.iloc[0]
                    if isinstance(first_value, str):
                        new_dtypes.iloc[i] = ArrowStringDtype()
                else:  # pragma: no cover
                    # empty, set arrow string dtype
                    new_dtypes.iloc[i] = ArrowStringDtype()
            else:
                # empty, set arrow string dtype
                new_dtypes.iloc[i] = ArrowStringDtype()
    return new_dtypes


def make_dtype(dtype):
    if isinstance(dtype, (np.dtype, ExtensionDtype)):
        return dtype
    return np.dtype(dtype) if dtype is not None else None


def make_dtypes(dtypes):
    if dtypes is None:
        return None
    if not isinstance(dtypes, pd.Series):
        dtypes = pd.Series(dtypes)
    return dtypes.apply(make_dtype)


def is_dataframe(x):
    if cudf is not None:  # pragma: no cover
        if isinstance(x, cudf.DataFrame):
            return True
    return isinstance(x, pd.DataFrame)


def is_series(x):
    if cudf is not None:  # pragma: no cover
        if isinstance(x, cudf.Series):
            return True
    return isinstance(x, pd.Series)


def is_index(x):
    if cudf is not None:  # pragma: no cover
        if isinstance(x, cudf.Index):
            return True
    return isinstance(x, pd.Index)


def get_xdf(x):
    if cudf is not None:  # pragma: no cover
        if isinstance(x, (cudf.DataFrame, cudf.Series, cudf.Index)):
            return cudf
    return pd


def is_cudf(x):
    if cudf is not None:  # pragma: no cover
        if isinstance(x, (cudf.DataFrame, cudf.Series, cudf.Index)):
            return True
    return False


def whether_to_clean_up(op, threshold):
    func = op.func
    counted_bytes = 0
    max_recursion_depth = 2

    from collections import deque
    from numbers import Number

    BYPASS_CLASSES = (str, bytes, Number, range, bytearray, pd.DataFrame, pd.Series)

    class GetSizeEarlyStopException(Exception):
        pass

    def check_exceed_threshold():
        nonlocal threshold, counted_bytes
        if counted_bytes >= threshold:
            raise GetSizeEarlyStopException()

    def getsize(obj_outer):
        _seen_obj_ids = set()

        def inner_count(obj, recursion_depth):
            obj_id = id(obj)
            if obj_id in _seen_obj_ids or recursion_depth > max_recursion_depth:
                return 0
            _seen_obj_ids.add(obj_id)
            recursion_depth += 1
            size = sys.getsizeof(obj)
            if isinstance(obj, BYPASS_CLASSES):
                return size
            elif isinstance(obj, (tuple, list, set, deque)):
                size += sum(inner_count(i, recursion_depth) for i in obj)
            elif hasattr(obj, "items"):
                size += sum(
                    inner_count(k, recursion_depth) + inner_count(v, recursion_depth)
                    for k, v in getattr(obj, "items")()
                )
            if hasattr(obj, "__dict__"):
                size += inner_count(vars(obj), recursion_depth)
            if hasattr(obj, "__slots__"):
                size += sum(
                    inner_count(getattr(obj, s), recursion_depth)
                    for s in obj.__slots__
                    if hasattr(obj, s)
                )
            return size

        return inner_count(obj_outer, 0)

    try:
        # Note: In most cases, func is just a function with closure, while chances are that
        # func is a callable that doesn't have __closure__ attribute.
        if inspect.isclass(func):
            pass
        elif hasattr(func, "__closure__") and func.__closure__ is not None:
            for cell in func.__closure__:
                counted_bytes += getsize(cell.cell_contents)
                check_exceed_threshold()
        elif callable(func):
            if hasattr(func, "__dict__"):
                for k, v in func.__dict__.items():
                    counted_bytes += sum([getsize(k), getsize(v)])
                    check_exceed_threshold()
            if hasattr(func, "__slots__"):
                for slot in func.__slots__:
                    counted_bytes += (
                        getsize(getattr(func, slot)) if hasattr(func, slot) else 0
                    )
                    check_exceed_threshold()
    except GetSizeEarlyStopException:
        logger.debug("Func needs cleanup.")
        op.need_clean_up_func = True
    else:
        assert op.need_clean_up_func is False
        logger.debug("Func doesn't need cleanup.")

    return op.need_clean_up_func


def concat_on_columns(objs: List) -> Any:
    xdf = get_xdf(objs[0])
    # In cudf, concat with axis=1 and ignore_index=False by default behaves opposite to pandas.
    # Cudf would reset the index when axis=1 and ignore_index=False, which does not match with its document.
    # Therefore, we deal with this case specially.
    result = xdf.concat(objs, axis=1)
    if xdf is cudf:
        result.index = objs[0].index
    return result


def patch_sa_engine_execute():
    """
    pandas did not resolve compatibility issue of sqlalchemy 2.0, the issue
    is https://github.com/pandas-dev/pandas/issues/40686. We need to patch
    Engine class in SQLAlchemy, and then our code can work well.
    """
    try:
        from sqlalchemy.engine import Engine
    except ImportError:  # pragma: no cover
        return

    def execute(self, statement, *multiparams, **params):
        connection = self.connect()
        return connection.execute(statement, *multiparams, **params)

    if hasattr(Engine, "execute"):  # pragma: no cover
        return
    Engine.execute = execute


def pack_func_args(df, funcs, *args, **kwargs) -> Any:
    """
    Pack the funcs with args and kwargs to avoid the ambiguity between other
    positional and keyword arguments. It will process the funcs by the following rule:

    1. If there's no such args and kwargs, return funcs itself.

    2. If the funcs is a dict-like object, it will iterate each key-value pair, pack the
    value recursively, and return a new dict with the same keys and packed values.

    3. If the funcs is a list-like object, it will iterate each element, pack it
    recursively, and return a new list with the packed elements.

    4. If the funcs is a str object, it will try to get the attribute df.funcs firstly,
    if it exists and is a callable, return a partial one with args and kwargs packed in.
    If it exists but isn't a callable, a ValueError is raised. If it doesn't exist, then
    try to get the attribute of np.funcs, if it exists and df is acceptable by funcs,
    return a partial one with args and kwargs packed in, otherwise an AttributeValue is
    raised. This rule is almost the same with pandas.

    5. Other cases are treated as funcs being a callable, returns the partial one with
    args and kwargs packed in.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        The DataFrame or Series object to test the function.
    funcs : function, str, list-like or dict-like
        Function to pack. It should have the same type with Dataframe.transform().
    *args :
        The positional arguments to func. If funcs contains many functions, each one
        should be able to accept *args.
    **kwargs :
        The keyword arguments to func. If funcs contains many functions, each one
        should be able to accept **kwargs.

    Returns
    -------
    The packed functions having the same structure with funcs.

    Raises
    ------
    ValueError :
        If there's a string but the corresponding function doesn't accept any positional
        or keyword arguments.
    AttributeError :
        If there's a string but no corresponding function is found.
    """
    if not args and not kwargs:
        return funcs

    if is_dict_like(funcs):
        return {k: pack_func_args(df, v, *args, **kwargs) for k, v in funcs.items()}

    if is_list_like(funcs):
        return [pack_func_args(df, v, *args, **kwargs) for v in funcs]

    f = get_callable_by_name(df, funcs) if isinstance(funcs, str) else funcs

    # Callable
    return functools.partial(f, *args, **kwargs)


def get_callable_by_name(df: Any, func_name: str) -> Callable:
    """
    Get the callable by the func name.
    It will try to get the attribute df.funcs firstly, if it exists and is a callable,
    return it. If it exists but isn't a callable, a ValueError is raised. If it doesn't
    exist, then try to get the attribute of np.funcs, if it exists and df is acceptable
    by funcs, return a partial one with args and kwargs packed in, otherwise an
    AttributeValue is raised. This rule is almost the same with pandas.

    Parameters
    ----------
    df: padnas.Series or pandas.Dataframe
        The receiver of the func name.
    func_name : str
        The func name.

    Returns
    -------
    The callable instance.

    Raises
    ------
    ValueError :
        If it's not a valid callable.
    AttributeError :
        If there's no corresponding function is found.
    """
    if hasattr(df, func_name):
        f = getattr(df, func_name)
        if callable(f):
            return f
        raise ValueError(f"{func_name} is not a callable")

    if hasattr(np, func_name) and hasattr(df, "__array__"):
        return getattr(np, func_name)

    raise AttributeError(
        f"'{func_name}' is not a valid function for '{type(df).__name__}' object"
    )
