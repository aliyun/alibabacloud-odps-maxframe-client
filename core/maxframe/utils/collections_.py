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

from collections import OrderedDict
from typing import Callable, List, Tuple, Type, Union

import pandas as pd


class AttributeDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttributeDict' object has no attribute {item}")


def find_objects(
    nested: Union[List, dict],
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


def replace_objects(nested: Union[List, dict], mapping) -> Union[List, dict]:
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


def is_empty(val):
    if isinstance(val, (pd.DataFrame, pd.Series, pd.Index)):
        return val.empty
    try:
        import numpy as np

        if isinstance(val, np.ndarray):
            return val.size == 0
    except ImportError:
        pass
    return not bool(val)


def flatten(nested_iterable: Union[List, tuple]) -> List:
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


def stack_back(flattened: List, raw: Union[List, tuple]) -> Union[List, tuple]:
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

    def _stack(items):
        result_items = []
        for item in items:
            if not isinstance(item, (list, tuple)):
                result_items.append(next(flattened_iter))
            else:
                nested_result = _stack(item)
                # Preserve the type of the nested container
                if isinstance(item, tuple):
                    result_items.append(tuple(nested_result))
                else:
                    result_items.append(nested_result)

        return result_items

    result = _stack(raw)
    # Preserve the type of the root container
    if isinstance(raw, tuple):
        return tuple(result)
    else:
        return result


class LRUDict(OrderedDict):
    def __init__(self, maxsize=16, *args, **kwargs):
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]
