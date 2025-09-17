import copy
import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from ....lib.version import parse as parse_version
from ....utils import deserialize_serializable, serialize_serializable, tokenize
from ...wrapped_pickle import switch_unpickle
from .. import ArrowDtype
from ..blob import ArrowBlobType, ExternalBlobDtype, SolidBlob


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
    try:
        assert pa_arr.tolist() == [b"abcd", b"efgh"]
    except NotImplementedError:
        # compatibility test for arrow 1.0
        assert pa_arr.storage.tolist() == [b"abcd", b"efgh"]

    if parse_version(pa.__version__) >= parse_version("3.0"):
        to_pd_ser = pa_arr.to_pandas()
        assert to_pd_ser.dtype == ExternalBlobDtype()
        pd.testing.assert_series_equal(pd_ser, to_pd_ser)
