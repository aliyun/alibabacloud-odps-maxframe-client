import contextlib
import copy
import pickle
import threading
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from maxframe.lib.dtypes_extension import ArrowDtype, blob
from maxframe.lib.dtypes_extension.blob import (
    ArrowBlobType,
    ExternalBlobDtype,
    SolidBlob,
    make_polars_blob_type,
    polars_to_pandas_with_blobs,
)
from maxframe.lib.version import parse as parse_version
from maxframe.lib.wrapped_pickle import switch_unpickle
from maxframe.utils import deserialize_serializable, serialize_serializable, tokenize

try:
    import polars as pl
except ImportError:  # pragma: no cover - polars is a hard test dependency
    pl = None

requires_polars = pytest.mark.skipif(
    pl is None or not hasattr(pl, "BaseExtension"),
    reason="need polars with BaseExtension support",
)


@switch_unpickle
def test_blob_object():
    blob = SolidBlob(b"text_content")
    assert blob == SolidBlob(b"text_content")
    assert blob == copy.copy(blob)
    assert blob == deserialize_serializable(serialize_serializable(blob))
    assert hash(blob) == hash(blob)
    assert tokenize(blob) == tokenize(blob)
    assert blob != b"other_content"

    with blob.open("rb") as reader:
        assert reader.read() == b"text_content"

    blob = SolidBlob()
    with pytest.raises(ValueError), blob.open("wb"):
        raise ValueError
    with blob.open("wb") as writer:
        writer.write(b"text_content")
    assert blob.reference == b"text_content"
    with switch_unpickle(forbidden=False):
        assert pickle.loads(pickle.dumps(blob)) == blob


@switch_unpickle
def test_blob_series():
    with pytest.raises(ValueError):
        pd.Series(np.array([["a", "b"], ["c", "d"]]), dtype="blob")

    pd_ser = pd.Series(["abcd", "efgh", "ijkl"], dtype="blob")
    assert tokenize(pd_ser) == tokenize(pd_ser)
    assert pd_ser[0].reference == b"abcd"
    pd.testing.assert_series_equal(pd_ser, pd_ser.copy(deep=True))
    pd.testing.assert_series_equal(
        pd_ser, deserialize_serializable(serialize_serializable(pd_ser))
    )

    part_ser = pd_ser.iloc[np.array([0, 1])]
    pd.testing.assert_series_equal(
        part_ser, pd.Series(np.array([b"abcd", b"efgh"], dtype="O"), dtype="blob")
    )
    ix = pd.Index([0, 2, 3])
    part_ser = pd_ser.reindex(ix)
    pd.testing.assert_series_equal(
        part_ser, pd.Series([b"abcd", b"ijkl", None], index=ix, dtype="blob")
    )

    pd_ser2 = pd.Series(["abcd"], dtype="blob")
    cat_ser = pd.concat([pd_ser, pd_ser2], ignore_index=True)
    pd.testing.assert_series_equal(
        cat_ser, pd.Series([b"abcd", b"efgh", b"ijkl", b"abcd"], dtype="blob")
    )

    cat_ser_bin = cat_ser.astype(ArrowDtype(pa.binary()))
    pd.testing.assert_series_equal(
        cat_ser_bin,
        pd.Series([b"abcd", b"efgh", b"ijkl", b"abcd"], dtype=ArrowDtype(pa.binary())),
    )


def test_blob_arrow_conversion():
    pd_ser = pd.Series([SolidBlob(b"abcd"), SolidBlob(b"efgh")], dtype="blob")
    pa_arr = pa.Array.from_pandas(pd_ser)
    assert pa_arr.type == ArrowBlobType()
    assert pa_arr.type == deserialize_serializable(serialize_serializable(pa_arr.type))
    try:
        assert pa_arr.tolist() == [b"abcd", b"efgh"]
    except NotImplementedError:
        # compatibility test for arrow 1.0
        assert pa_arr.storage.tolist() == [b"abcd", b"efgh"]

    if parse_version(pa.__version__) >= parse_version("3.0"):
        to_pd_ser = pa_arr.to_pandas()
        assert to_pd_ser.dtype == ExternalBlobDtype()
        pd.testing.assert_series_equal(pd_ser, to_pd_ser)


def test_blob_null_conversion():
    # A blob series containing None must round-trip through both the Arrow
    # conversion (__arrow_array__) and astype(ArrowDtype(pa.binary())) without
    # dereferencing the None slot.
    pd_ser = pd.Series([SolidBlob(b"abcd"), None, SolidBlob(b"ijkl")], dtype="blob")
    assert pd_ser.isna().tolist() == [False, True, False]

    # __arrow_array__: None -> Arrow null
    pa_arr = pa.Array.from_pandas(pd_ser)
    try:
        assert pa_arr.tolist() == [b"abcd", None, b"ijkl"]
    except NotImplementedError:
        assert pa_arr.storage.tolist() == [b"abcd", None, b"ijkl"]

    # astype(ArrowDtype(pa.binary())): None preserved, blobs read out
    bin_ser = pd_ser.astype(ArrowDtype(pa.binary()))
    pd.testing.assert_series_equal(
        bin_ser,
        pd.Series([b"abcd", None, b"ijkl"], dtype=ArrowDtype(pa.binary())),
    )

    # __from_arrow__: null slots restore as None (not SolidBlob(None)), and
    # non-null slots read back their original bytes.  This closes the Arrow
    # round-trip so a blob column survives to_arrow()/from_arrow() intact.
    # to_pylist() on an ExtensionArray requires pyarrow>=3.0 (legacy 1.0
    # raises ArrowNotImplementedError), matching the guard in
    # test_blob_arrow_conversion.
    if parse_version(pa.__version__) >= parse_version("3.0"):
        restored = ExternalBlobDtype().__from_arrow__(pa_arr)
        assert restored.isna().tolist() == [False, True, False]
        with restored[0].open("rb") as reader:
            assert reader.read() == b"abcd"
        assert restored[1] is None


# ---------------------------------------------------------------------------
# Concurrent conversion of one shared Polars frame (bug #87341179)
# ---------------------------------------------------------------------------
# ``to_pandas()`` on a frame from several threads deadlocks inside Polars - the
# deadlock depends on the Polars build and the frame shape, so it cannot be
# asserted on directly.  Its precondition is pinned down instead: the
# conversion body of one frame never runs in two threads at once.

_ROWS = 50
_THREADS = 4
_ITERS = 10


def _make_polars_blob_frame():
    """A Polars frame with regular columns plus a blob column."""
    return pl.DataFrame(
        {
            "int_col": list(range(_ROWS)),
            "str_col": ["s%d" % (i % 10) for i in range(_ROWS)],
            "blob_col": pl.Series(
                "blob_col",
                [SolidBlob(b"blob-%d" % i) for i in range(_ROWS)],
                dtype=make_polars_blob_type(),
            ),
        }
    )


def _convert_in_threads(frame):
    """Convert ``frame`` from several threads, returning every converted frame."""
    pdfs = []

    def worker():
        pdfs.extend(polars_to_pandas_with_blobs(frame) for _ in range(_ITERS))

    workers = [threading.Thread(target=worker) for _ in range(_THREADS)]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()

    return pdfs


@requires_polars
@pytest.mark.parametrize("serialised", [True, False], ids=["locked", "unlocked"])
def test_shared_frame_conversions_are_serialised(monkeypatch, serialised):
    """The conversion body must never run in two threads at once for one frame.

    ``unlocked`` replaces the frame lock with a no-op, which is the pre-fix
    behaviour: the detector must report an overlap then, otherwise the clean
    ``locked`` run over the same detector would prove nothing.
    """
    convert = blob._polars_to_pandas_with_blobs  # captured before it is wrapped
    state_lock = threading.Lock()
    active = []
    overlaps = []

    def convert_with_detector(pl_df):
        with state_lock:
            if active:
                overlaps.append(1)
            active.append(1)
        try:
            time.sleep(0.002)  # widen the window in which an overlap is visible
            return convert(pl_df)
        finally:
            with state_lock:
                active.pop()

    monkeypatch.setattr(blob, "_polars_to_pandas_with_blobs", convert_with_detector)
    if not serialised:
        monkeypatch.setattr(
            blob, "polars_frame_lock", lambda frame: contextlib.nullcontext()
        )

    pdfs = _convert_in_threads(_make_polars_blob_frame())

    assert len(pdfs) == _THREADS * _ITERS
    assert bool(overlaps) is not serialised


@requires_polars
def test_concurrent_conversion_matches_serial_result():
    """Concurrency must not change the converted frame, blob column included."""
    frame = _make_polars_blob_frame()
    expected = polars_to_pandas_with_blobs(frame)

    pdfs = _convert_in_threads(frame)

    assert len(pdfs) == _THREADS * _ITERS
    for pdf in pdfs:
        assert pdf["int_col"].tolist() == expected["int_col"].tolist()
        assert pdf["blob_col"].tolist() == expected["blob_col"].tolist()
        assert pdf["blob_col"].dtype == expected["blob_col"].dtype
