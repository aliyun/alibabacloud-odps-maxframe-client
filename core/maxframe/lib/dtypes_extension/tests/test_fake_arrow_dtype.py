import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pandas.api.types import pandas_dtype

from maxframe.lib.version import parse as parse_version
from maxframe.lib.wrapped_pickle import switch_unpickle
from maxframe.utils import deserialize_serializable, serialize_serializable, tokenize

try:
    from pandas import ArrowDtype  # noqa: F401

    ArrowDtype(pa.string())
    pytestmark = pytest.mark.skip("Only test when ArrowDtype not available in pandas")
except ImportError:
    from maxframe.lib.dtypes_extension._fake_arrow_dtype import (
        FakeArrowDtype,
        FakeArrowExtensionArray,
        to_pyarrow_type,
    )


def _comparison_result_to_pylist(result):
    if hasattr(result, "_pa_array"):
        return result._pa_array.to_pylist()
    return [None if pd.isna(v) else bool(v) for v in result.tolist()]


def test_fake_arrow_dtype():
    assert to_pyarrow_type(np.dtype("int64")) == pa.int64()
    assert to_pyarrow_type(pa.string()) == pa.string()
    assert to_pyarrow_type(FakeArrowDtype(pa.string())) == pa.string()
    assert to_pyarrow_type(FakeArrowDtype(pa.bool_())) == pa.bool_()
    assert to_pyarrow_type(FakeArrowDtype(pa.int8())) == pa.int8()

    pd_type = pandas_dtype("binary[pyarrow]")
    assert isinstance(pd_type, FakeArrowDtype)
    assert pd_type.pyarrow_dtype == pa.binary()


@switch_unpickle
def test_arrow_series():
    if parse_version(pa.__version__).major < 2:
        pytest.skip("pyarrow need to be >= 2.0 to run this case")

    empty_pd_ser = pd.Series(np.array([]), dtype=FakeArrowDtype(pa.binary()))
    assert len(empty_pd_ser) == 0

    pd_ser = pd.Series([b"abcd", b"efgh", b"ijkl"], dtype=FakeArrowDtype(pa.binary()))
    assert tokenize(pd_ser) == tokenize(pd_ser)
    assert pd_ser[0] == b"abcd"
    pd.testing.assert_series_equal(pd_ser, pd_ser.copy(deep=True))
    pd.testing.assert_series_equal(
        pd_ser, deserialize_serializable(serialize_serializable(pd_ser))
    )
    part_ser = pd_ser.iloc[np.array([0, 1])]
    pd.testing.assert_series_equal(
        part_ser,
        pd.Series(
            np.array([b"abcd", b"efgh"], dtype="O"), dtype=FakeArrowDtype(pa.binary())
        ),
    )
    ix = pd.Index([0, 2, 3])
    part_ser = pd_ser.reindex(ix)
    pd.testing.assert_series_equal(
        part_ser,
        pd.Series(
            [b"abcd", b"ijkl", None], index=ix, dtype=FakeArrowDtype(pa.binary())
        ),
    )
    pd_ser2 = pd.Series([b"abcd"], dtype=FakeArrowDtype(pa.binary()))
    cat_ser = pd.concat([pd_ser, pd_ser2], ignore_index=True)
    pd.testing.assert_series_equal(
        cat_ser,
        pd.Series(
            [b"abcd", b"efgh", b"ijkl", b"abcd"], dtype=FakeArrowDtype(pa.binary())
        ),
    )
    cat_ser_bin = cat_ser.astype(FakeArrowDtype(pa.binary()))
    pd.testing.assert_series_equal(
        cat_ser_bin,
        pd.Series(
            [b"abcd", b"efgh", b"ijkl", b"abcd"], dtype=FakeArrowDtype(pa.binary())
        ),
    )


def test_fake_arrow_index_drop_uses_equals():
    """FakeArrow-backed Index must survive pandas schema operations.

    Reproduces the real failure: framedriver deserializes a pandas-3
    string-typed dtypes Index as FakeArrow-backed.  Series.drop() then
    triggers Index.equals -> ExtensionArray.__eq__, which was missing.
    """
    arr = FakeArrowExtensionArray(pa.array(["uid", "txt", "tmp"], type=pa.string()))
    s = pd.Series(["int64", "object", "int64"], index=pd.Index(arr))

    # drop([]) / drop(missing) trigger Index.equals on unchanged index
    pd.testing.assert_series_equal(s.drop([], errors="ignore"), s)
    pd.testing.assert_series_equal(s.drop(["grp"], errors="ignore"), s)
    assert list(s.drop(["uid"], errors="ignore").index) == ["txt", "tmp"]


def test_fake_arrow_array_eq_preserves_nulls():
    with_null = FakeArrowExtensionArray(
        pa.array(["uid", "txt", None], type=pa.string())
    )
    same = FakeArrowExtensionArray(pa.array(["uid", "txt", None], type=pa.string()))

    assert (with_null == same)._pa_array.to_pylist() == [True, True, None]
    assert (with_null == "uid")._pa_array.to_pylist() == [True, False, None]
    assert ("uid" == with_null)._pa_array.to_pylist() == [True, False, None]
    assert (with_null == 1)._pa_array.to_pylist() == [False, False, None]
    assert (with_null == pd.NA)._pa_array.to_pylist() == [None, None, None]
    assert (with_null == np.nan)._pa_array.to_pylist() == [None, None, None]
    assert (
        with_null == np.array(["uid", "tmp", None], dtype=object)
    )._pa_array.to_pylist() == [True, False, None]
    assert (with_null == ["uid", pd.NA, None])._pa_array.to_pylist() == [
        True,
        None,
        None,
    ]
    assert (
        with_null == np.array(["uid", pd.NA, None], dtype=object)
    )._pa_array.to_pylist() == [True, None, None]


def test_fake_arrow_array_ne_preserves_nulls():
    with_null = FakeArrowExtensionArray(
        pa.array(["uid", "txt", None], type=pa.string())
    )
    other = FakeArrowExtensionArray(pa.array(["uid", "tmp", None], type=pa.string()))

    assert (with_null != other)._pa_array.to_pylist() == [False, True, None]
    assert (with_null != "uid")._pa_array.to_pylist() == [False, True, None]
    assert ("uid" != with_null)._pa_array.to_pylist() == [False, True, None]
    assert (with_null != 1)._pa_array.to_pylist() == [True, True, None]
    assert (with_null != pd.NA)._pa_array.to_pylist() == [None, None, None]
    assert (with_null != np.nan)._pa_array.to_pylist() == [None, None, None]
    assert (
        with_null != np.array(["uid", "tmp", None], dtype=object)
    )._pa_array.to_pylist() == [False, True, None]
    assert (with_null != ["uid", pd.NA, None])._pa_array.to_pylist() == [
        False,
        None,
        None,
    ]
    assert (
        with_null != np.array(["uid", pd.NA, None], dtype=object)
    )._pa_array.to_pylist() == [False, None, None]
    index_result = _comparison_result_to_pylist(pd.Index(with_null) != pd.Index(other))
    assert index_result[:2] == [False, True]
    assert len(index_result) == 3


def test_fake_arrow_array_equals_checks_arrow_storage():
    arr = FakeArrowExtensionArray(pa.array(["uid", "txt", "tmp"], type=pa.string()))

    assert (
        arr.equals(
            FakeArrowExtensionArray(pa.array(["uid", "txt", "tmp"], type=pa.string()))
        )
        is True
    )
    assert (
        arr.equals(
            FakeArrowExtensionArray(
                pa.array(["uid", "txt", "tmp"], type=pa.large_string())
            )
        )
        is False
    )
    assert (
        arr.equals(
            FakeArrowExtensionArray(pa.array(["uid", "tmp", "x"], type=pa.string()))
        )
        is False
    )
    assert (
        arr.equals(
            FakeArrowExtensionArray(
                pa.chunked_array(
                    [
                        pa.array(["uid"], type=pa.string()),
                        pa.array(["txt", "tmp"], type=pa.string()),
                    ]
                )
            )
        )
        is True
    )
