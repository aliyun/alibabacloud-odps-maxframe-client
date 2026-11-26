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

"""Tests for :mod:`maxframe.utils.polars`.

The per-frame lock used by ``to_pandas()`` conversions: converting the *same*
frame from several chunk threads deadlocks inside Polars.  Plus the ``(N, 0)``
capability gate and its carrier builder.
"""

import gc
import threading
import time
import weakref

import pytest

from maxframe.utils import polars as polars_utils
from maxframe.utils.polars import (
    polars_frame_lock,
    polars_supports_zero_width_height,
    zero_width_frame,
)

try:
    import polars as pl
except ImportError:  # pragma: no cover - polars is a hard test dependency
    pl = None

pytestmark = pytest.mark.skipif(
    pl is None or not hasattr(pl, "BaseExtension"),
    reason="need polars with BaseExtension support",
)


@pytest.fixture
def empty_registry():
    """Keep the module-level lock registry from leaking between tests."""
    polars_utils._FRAME_LOCKS.clear()
    yield
    polars_utils._FRAME_LOCKS.clear()


def _frame_of_kind(kind):
    """A frame of the given kind; every Polars frame type is lockable."""
    frame = pl.DataFrame({"a": [1, 2, 3]})
    return {"dataframe": frame, "lazyframe": frame.lazy(), "series": frame["a"]}[kind]


@pytest.mark.parametrize("kind", ["dataframe", "lazyframe", "series"])
def test_lock_is_per_frame_instance(empty_registry, kind):
    """One frame object gets one lock; distinct frames never share a lock."""
    frame = _frame_of_kind(kind)
    assert polars_frame_lock(frame) is polars_frame_lock(frame)
    assert polars_frame_lock(frame) is not polars_frame_lock(_frame_of_kind(kind))


def test_lock_is_reentrant(empty_registry):
    """A guarded call may itself make a guarded call on the same frame."""
    frame = pl.DataFrame({"a": [1]})
    lock = polars_frame_lock(frame)
    with lock:
        # Re-entering the same frame's lock must succeed.
        assert polars_frame_lock(frame).acquire(blocking=False)
        lock.release()


def test_holders_of_one_frame_exclude_each_other(empty_registry):
    """Two threads must never be inside one frame's lock at the same time."""
    frame = pl.DataFrame({"a": [1]})
    state_lock = threading.Lock()
    active = []
    overlaps = []

    def hold():
        for _ in range(40):
            with polars_frame_lock(frame):
                with state_lock:
                    if active:
                        overlaps.append(1)
                    active.append(1)
                time.sleep(0.001)  # widen the window where an overlap is visible
                with state_lock:
                    active.pop()

    threads = [threading.Thread(target=hold) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert overlaps == []


def test_collected_frames_drop_their_locks(empty_registry):
    """The registry must not pile up entries for frames that are gone."""
    frame = pl.DataFrame({"a": [1]})
    collected = weakref.ref(frame)
    polars_frame_lock(frame)
    del frame
    gc.collect()

    assert collected() is None
    assert polars_utils._FRAME_LOCKS == {}


def test_reused_id_never_hands_out_a_dead_frames_lock(empty_registry):
    """The registry is keyed by ``id(frame)``, so a stale entry must not match.

    A stale entry survives only until the collected frame's weakref callback
    runs; the allocator reissuing its id within that window looks exactly like
    the dead frame's entry sitting under the new frame's key.
    """
    dead = pl.DataFrame({"a": [1]})
    replacement = pl.DataFrame({"a": [2]})
    dead_lock = polars_frame_lock(dead)
    entry = polars_utils._FRAME_LOCKS.pop(id(dead))
    del dead
    gc.collect()

    polars_utils._FRAME_LOCKS[id(replacement)] = entry

    assert polars_frame_lock(replacement) is not dead_lock
    # ... and the replacement keeps a lock of its own from then on.
    assert polars_frame_lock(replacement) is polars_frame_lock(replacement)


# ---------------------------------------------------------------------------
# (N, 0) capability gate
# ---------------------------------------------------------------------------


@pytest.fixture
def legacy_zero_width(monkeypatch):
    """Force the pre-1.38 behaviour (no zero-column height support)."""
    monkeypatch.setattr(polars_utils, "_POLARS_SUPPORTS_ZERO_WIDTH_HEIGHT", False)


def test_zero_width_frame_carries_height():
    if not polars_supports_zero_width_height():  # pragma: no cover
        pytest.skip("polars build without (N, 0) support")
    assert zero_width_frame(3).shape == (3, 0)


def test_zero_width_frame_legacy_has_no_height(legacy_zero_width):
    assert zero_width_frame(3).shape == (0, 0)


def test_zero_width_frame_without_height_is_empty():
    # A falsy height returns the empty frame before support is consulted, so
    # this holds on both build classes.
    assert zero_width_frame(None).shape == (0, 0)
    assert zero_width_frame(0).shape == (0, 0)
