import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pandas.api.types import pandas_dtype

from ....lib.version import parse as parse_version
from ....utils import deserialize_serializable, serialize_serializable, tokenize
from ...wrapped_pickle import switch_unpickle

try:
    from pandas import ArrowDtype  # noqa: F401

    pytestmark = pytest.mark.skip("Only test when ArrowDtype not available in pandas")
except ImportError:
    from .._fake_arrow_dtype import FakeArrowDtype, to_pyarrow_type


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
