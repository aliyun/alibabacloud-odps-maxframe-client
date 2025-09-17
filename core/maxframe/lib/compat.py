import asyncio
import functools
from typing import TYPE_CHECKING, Callable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas._typing import ArrayLike, Scalar


def case_when(
    self,
    caselist: List[
        Tuple[
            Union[
                "ArrayLike",
                Callable[[pd.Series], Union[pd.Series, np.ndarray, Sequence[bool]]],
            ],
            Union[
                "ArrayLike",
                "Scalar",
                Callable[[pd.Series], Union[pd.Series, np.ndarray]],
            ],
        ],
    ],
) -> pd.Series:
    """
    Replace values where the conditions are True.

    Parameters
    ----------
    caselist : A list of tuples of conditions and expected replacements
        Takes the form:  ``(condition0, replacement0)``,
        ``(condition1, replacement1)``, ... .
        ``condition`` should be a 1-D boolean array-like object
        or a callable. If ``condition`` is a callable,
        it is computed on the Series
        and should return a boolean Series or array.
        The callable must not change the input Series
        (though pandas doesn`t check it). ``replacement`` should be a
        1-D array-like object, a scalar or a callable.
        If ``replacement`` is a callable, it is computed on the Series
        and should return a scalar or Series. The callable
        must not change the input Series
        (though pandas doesn`t check it).

        .. versionadded:: 2.2.0

    Returns
    -------
    Series

    See Also
    --------
    Series.mask : Replace values where the condition is True.

    Examples
    --------
    >>> c = pd.Series([6, 7, 8, 9], name='c')
    >>> a = pd.Series([0, 0, 1, 2])
    >>> b = pd.Series([0, 3, 4, 5])

    >>> c.case_when(caselist=[(a.gt(0), a),  # condition, replacement
    ...                       (b.gt(0), b)])
    0    6
    1    3
    2    1
    3    2
    Name: c, dtype: int64
    """
    from pandas.api.types import is_scalar
    from pandas.core import common as com
    from pandas.core.construction import array as pd_array
    from pandas.core.dtypes.cast import (
        construct_1d_arraylike_from_scalar,
        find_common_type,
        infer_dtype_from,
    )
    from pandas.core.dtypes.generic import ABCSeries

    if not isinstance(caselist, list):
        raise TypeError(
            f"The caselist argument should be a list; instead got {type(caselist)}"
        )

    if not caselist:
        raise ValueError(
            "provide at least one boolean condition, "
            "with a corresponding replacement."
        )

    for num, entry in enumerate(caselist):
        if not isinstance(entry, tuple):
            raise TypeError(
                f"Argument {num} must be a tuple; instead got {type(entry)}."
            )
        if len(entry) != 2:
            raise ValueError(
                f"Argument {num} must have length 2; "
                "a condition and replacement; "
                f"instead got length {len(entry)}."
            )
    caselist = [
        (
            com.apply_if_callable(condition, self),
            com.apply_if_callable(replacement, self),
        )
        for condition, replacement in caselist
    ]
    default = self.copy()
    conditions, replacements = zip(*caselist)
    common_dtypes = [infer_dtype_from(arg)[0] for arg in [*replacements, default]]
    if len(set(common_dtypes)) > 1:
        common_dtype = find_common_type(common_dtypes)
        updated_replacements = []
        for condition, replacement in zip(conditions, replacements):
            if is_scalar(replacement):
                replacement = construct_1d_arraylike_from_scalar(
                    value=replacement, length=len(condition), dtype=common_dtype
                )
            elif isinstance(replacement, ABCSeries):
                replacement = replacement.astype(common_dtype)
            else:
                replacement = pd_array(replacement, dtype=common_dtype)
            updated_replacements.append(replacement)
        replacements = updated_replacements
        default = default.astype(common_dtype)

    counter = reversed(range(len(conditions)))
    for position, condition, replacement in zip(
        counter, conditions[::-1], replacements[::-1]
    ):
        try:
            default = default.mask(
                condition, other=replacement, axis=0, inplace=False, level=None
            )
        except Exception as error:
            raise ValueError(
                f"Failed to apply condition{position} and replacement{position}."
            ) from error
    return default


def patch_pandas():
    if not hasattr(pd.Series, "case_when"):
        pd.Series.case_when = case_when


class cached_property:
    """
    A property that is only computed once per instance and then replaces itself
    with an ordinary attribute. Deleting the attribute resets the property.
    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """  # noqa

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self

        if asyncio.iscoroutinefunction(self.func):
            return self._wrap_in_coroutine(obj)

        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

    def _wrap_in_coroutine(self, obj):
        @functools.wraps(obj)
        def wrapper():
            future = asyncio.ensure_future(self.func(obj))
            obj.__dict__[self.func.__name__] = future
            return future

        return wrapper()


# isort: off
try:
    from functools import cached_property  # noqa: F811, F401
except ImportError:
    pass
