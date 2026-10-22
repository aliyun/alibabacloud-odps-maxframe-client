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

import inspect
from functools import wraps

from pandas.arrays import DatetimeArray, TimedeltaArray

from maxframe.utils import adapt_docstring

# dt method-style accessors delegate to ``*args, **kwargs`` wrappers on
# ``pd.Series.dt``; the real signatures live on the extension arrays.
# Map each to the class carrying the inspectable signature.
_DT_SIGNATURE_SOURCES = {
    "strftime": DatetimeArray,
    "day_name": DatetimeArray,
    "month_name": DatetimeArray,
    "normalize": DatetimeArray,
    "round": DatetimeArray,
    "floor": DatetimeArray,
    "ceil": DatetimeArray,
    "total_seconds": TimedeltaArray,
}


def _strip_examples(doc):
    """Drop the Examples section (header through end) from a numpy docstring."""
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "Examples":
            # Trim the underline (e.g. "--------") and any trailing blank lines.
            end = i
            while end > 0 and not lines[end - 1].strip():
                end -= 1
            return "\n".join(lines[:end]).rstrip()
    return doc


def _adapt_liteframe_docstring(doc):
    """Adapt a pandas accessor docstring for the LiteFrame accessor.

    Drops the pandas ``Examples`` section (LiteFrame examples are authored
    separately) and rewrites ``Series.`` cross-references to ``LiteFrame.``.
    """
    if doc is None:
        return None
    doc = _strip_examples(doc)
    doc = adapt_docstring(doc)
    return doc.replace("Series.", "LiteFrame.")


def _resolve_signature_source(accessor, method):
    """Return the callable whose signature a generated wrapper should expose.

    ``pd.Series.str.<method>`` carries its own signature, so it is returned
    directly.  ``pd.Series.dt.<method>`` is a ``*args, **kwargs`` delegate;
    for the method-style members the real signature lives on the extension
    array class recorded in ``_DT_SIGNATURE_SOURCES``.
    """
    source = getattr(accessor, method, None)
    if source is None:
        return None
    try:
        sig = inspect.signature(source)
        if "*args" not in str(sig) and "**kwargs" not in str(sig):
            return source
    except (TypeError, ValueError):
        pass
    cls = _DT_SIGNATURE_SOURCES.get(method)
    if cls is not None:
        return getattr(cls, method, None)
    return None


def _copy_signature(func, accessor, method):
    """Copy the pandas signature/name onto *func* in place; return *func*.

    ``functools.wraps`` would overwrite ``__doc__``; the caller has already
    adapted the docstring, so it is preserved across the wrap.
    """
    source = _resolve_signature_source(accessor, method)
    if source is None:
        return func
    doc = func.__doc__
    wrapped = wraps(source)(func)
    wrapped.__doc__ = doc
    return wrapped
