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

import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from maxframe.errors import OutputColumnMismatchError, OutputDtypeMismatchError
from maxframe.lib.dtypes_extension import ArrowBlobType, ArrowDtype, ExternalBlobDtype
from maxframe.utils.datatypes import (
    arrow_type_from_str,
    check_dtype_compatibility,
    is_arrow_dtype_supported,
    is_bool_dtype,
    is_datetime64_dtype,
    is_string_dtype,
    make_dtype,
    make_dtypes,
    validate_and_align_output,
    wrap_arrow_dtype,
)

try:
    from maxframe.lib.dtypes_extension.vector import ArrowVectorType
except ImportError:
    ArrowVectorType = None


def test_arrow_type_from_string():
    def _assert_arrow_type_convert(tp: pa.DataType) -> None:
        assert arrow_type_from_str(str(tp)) == tp

    _assert_arrow_type_convert(pa.int64())
    _assert_arrow_type_convert(pa.binary(20))
    _assert_arrow_type_convert(pa.decimal128(10, 2))
    _assert_arrow_type_convert(pa.list_(pa.int64()))
    _assert_arrow_type_convert(pa.map_(pa.string(), pa.int64()))
    _assert_arrow_type_convert(pa.date32())
    _assert_arrow_type_convert(pa.date64())
    _assert_arrow_type_convert(
        pa.struct([("key", pa.string()), ("value", pa.list_(pa.int64()))])
    )
    _assert_arrow_type_convert(
        pa.struct([("key", pa.string(), False), ("value", pa.list_(pa.int64()))])
    )
    _assert_arrow_type_convert(
        pa.struct([("key", pa.string()), ("value", pa.list_(pa.int64()), False)])
    )
    _assert_arrow_type_convert(ArrowBlobType())

    # Timezone-aware timestamps stringify with a ``tz=`` keyword that the
    # generic tokenizer must handle (including IANA names containing ``/``).
    _assert_arrow_type_convert(pa.timestamp("us"))
    _assert_arrow_type_convert(pa.timestamp("us", tz="UTC"))
    _assert_arrow_type_convert(pa.timestamp("ns", tz="America/New_York"))
    _assert_arrow_type_convert(pa.timestamp("s", tz="Asia/Shanghai"))
    # IANA identifiers may contain hyphens ("Port-au-Prince") as well as
    # slashes; the tokenizer must join both into a single name.
    _assert_arrow_type_convert(pa.timestamp("us", tz="America/Port-au-Prince"))
    _assert_arrow_type_convert(pa.timestamp("ns", tz="Asia/Ho_Chi_Minh"))

    # Dictionary types use ``key=value`` keyword syntax for values/indices/ordered.
    _assert_arrow_type_convert(pa.dictionary(pa.int32(), pa.string()))
    _assert_arrow_type_convert(pa.dictionary(pa.int8(), pa.int64(), ordered=True))
    _assert_arrow_type_convert(pa.dictionary(pa.int16(), pa.list_(pa.int64())))
    _assert_arrow_type_convert(pa.dictionary(pa.int32(), pa.timestamp("us", tz="UTC")))

    # large_list uses the same ``<item: T>`` form as list.
    _assert_arrow_type_convert(pa.large_list(pa.int64()))
    _assert_arrow_type_convert(pa.large_list(pa.list_(pa.int64())))

    # Keyword-argument types must also round-trip when nested inside other
    # parameterized types, not only at the top level.
    _assert_arrow_type_convert(pa.list_(pa.timestamp("us", tz="UTC")))
    _assert_arrow_type_convert(pa.list_(pa.dictionary(pa.int32(), pa.string())))
    _assert_arrow_type_convert(
        pa.struct(
            [
                ("ts", pa.timestamp("us", tz="UTC")),
                ("d", pa.dictionary(pa.int32(), pa.string())),
            ]
        )
    )
    _assert_arrow_type_convert(pa.large_list(pa.timestamp("ns", tz="America/New_York")))
    _assert_arrow_type_convert(
        pa.large_list(pa.timestamp("us", tz="America/Port-au-Prince"))
    )
    _assert_arrow_type_convert(pa.list_(pa.list_(pa.timestamp("us", tz="UTC"))))
    _assert_arrow_type_convert(
        pa.map_(pa.string(), pa.dictionary(pa.int8(), pa.int64(), ordered=True))
    )

    if ArrowVectorType is not None:
        _assert_arrow_type_convert(ArrowVectorType(pa.float32(), 128))
        _assert_arrow_type_convert(ArrowVectorType(pa.float64(), 256))


@pytest.mark.parametrize(
    "val, expected, make_series",
    [
        (int, np.dtype(int), True),
        ([int], [np.dtype(int)], False),
        ([int], pd.Series([np.dtype(int)]), True),
        (np.dtype("float64"), np.dtype("float64"), True),
        ("category", "category", True),
        ("string", "string", True),
        (pd.Timestamp, np.dtype("datetime64[ns]"), True),
        (pd.Timedelta, np.dtype("timedelta64[ns]"), True),
        ({"col": float}, {"col": np.dtype(float)}, False),
        ({"col": float}, pd.Series({"col": np.dtype(float)}), True),
        (
            pd.Series([float], index=["col"]),
            pd.Series([np.dtype(float)], index=["col"]),
            True,
        ),
    ],
)
def test_make_dtypes(val, expected, make_series):
    result = make_dtypes(val, make_series=make_series)
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "input_dtype, expected",
    [
        (wrap_arrow_dtype(pa.int64()), False),
        (wrap_arrow_dtype(pa.string()), True),
        (np.dtype("O"), True),
        (np.dtype("float32"), False),
        (pd.Int64Dtype(), False),
        (pd.StringDtype(), True),
    ],
)
def test_is_string_dtype(input_dtype, expected):
    assert is_string_dtype(input_dtype) is expected


@pytest.mark.parametrize(
    "input_dtype, expected",
    [
        (wrap_arrow_dtype(pa.int64()), False),
        (wrap_arrow_dtype(pa.bool_()), True),
        (np.dtype("bool"), True),
        (np.dtype("float32"), False),
        (pd.Int64Dtype(), False),
        (pd.BooleanDtype(), True),
    ],
)
def test_is_bool_dtype(input_dtype, expected):
    assert is_bool_dtype(input_dtype) is expected


@pytest.mark.parametrize(
    "input_dtype, expected",
    [
        (wrap_arrow_dtype(pa.int64()), False),
        (wrap_arrow_dtype(pa.timestamp("ms")), True),
        (np.dtype("bool"), False),
        (np.dtype("datetime64[ms]"), True),
        (pd.Int64Dtype(), False),
    ],
)
def test_is_datetime64_dtype(input_dtype, expected):
    assert is_datetime64_dtype(input_dtype) is expected


# New tests for previously untested functions
@pytest.mark.parametrize(
    "input_dtype, expected",
    [
        (int, np.dtype(int)),
        ("float64", np.dtype("float64")),
        (np.dtype("bool"), np.dtype("bool")),
        (pd.StringDtype(), pd.StringDtype()),
        (pd.BooleanDtype(), pd.BooleanDtype()),
        ("category", "category"),
        ("string", "string"),
        (pd.Timestamp, np.dtype("datetime64[ns]")),
        (pd.Timedelta, np.dtype("timedelta64[ns]")),
    ],
)
def test_make_dtype(input_dtype, expected):
    result = make_dtype(input_dtype)
    assert result == expected


def test_wrap_arrow_dtype():
    if not is_arrow_dtype_supported():
        pytest.skip("ArrowDtype is not supported")

    # Test wrapping various Arrow types
    wrapped_int = wrap_arrow_dtype(pa.int64())
    # wrap_arrow_dtype returns either an ArrowDtype or the raw Arrow type
    # The important thing is that it preserves the dtype information
    assert wrapped_int is not None

    wrapped_string = wrap_arrow_dtype(pa.string())
    assert isinstance(wrapped_string, pd.StringDtype)

    wrapped_timestamp = wrap_arrow_dtype(pa.timestamp("ms"))
    assert wrapped_timestamp is not None

    # The exact behavior depends on whether ArrowDtype is available
    if is_arrow_dtype_supported():
        # When ArrowDtype is supported, wrap_arrow_dtype may return an ArrowDtype
        # or a special wrapper that has a pyarrow_dtype attribute
        if hasattr(wrapped_int, "pyarrow_dtype"):
            assert wrapped_int.pyarrow_dtype == pa.int64()
        # Otherwise it might return the raw Arrow type
    else:
        # When ArrowDtype is not supported, should return the raw Arrow type
        assert wrapped_int == pa.int64()


def test_validate_and_align_output():
    """Test validate_and_align_output with various check_output_dtypes modes."""
    # Test 1: DataFrame with check_output_dtypes=False skips validation
    df = pd.DataFrame({"a": [1, 2, 3]})  # int64
    expected = pd.Series({"a": np.dtype("float64")})
    result = validate_and_align_output(df, expected, check_output_dtypes=False)
    assert result["a"].dtype == np.dtype("int64")

    # Test 2: DataFrame with check_output_dtypes='ignore' skips validation
    result = validate_and_align_output(df, expected, check_output_dtypes="ignore")
    assert result["a"].dtype == np.dtype("int64")

    # Test 3: DataFrame column check skipped when check_output_dtypes=False
    df_cols = pd.DataFrame({"b": [4, 5, 6], "a": [1, 2, 3]})
    expected_cols = pd.Series({"a": np.dtype("int64")})
    result = validate_and_align_output(
        df_cols, expected_cols, check_output_dtypes=False
    )
    assert list(result.columns) == ["b", "a"]

    # Test 4: DataFrame column check skipped when check_output_dtypes='ignore'
    result = validate_and_align_output(
        df_cols, expected_cols, check_output_dtypes="ignore"
    )
    assert list(result.columns) == ["b", "a"]

    # Test 5: Series with check_output_dtypes=False skips validation
    s = pd.Series([1, 2, 3])  # int64
    expected_s = np.dtype("float64")
    result = validate_and_align_output(s, expected_s, check_output_dtypes=False)
    assert result.dtype == np.dtype("int64")

    # Test 6: Series with check_output_dtypes='ignore' skips validation
    result = validate_and_align_output(s, expected_s, check_output_dtypes="ignore")
    assert result.dtype == np.dtype("int64")

    # Test 7: DataFrame with check_output_dtypes='warns' performs validation and casting
    result = validate_and_align_output(df, expected, check_output_dtypes="warns")
    assert result["a"].dtype == np.dtype("float64")

    # Test 8: DataFrame with check_output_dtypes='raises' raises on column mismatch
    df_missing = pd.DataFrame({"a": [1, 2, 3]})
    expected_missing = pd.Series({"a": np.dtype("int64"), "b": np.dtype("int64")})
    with pytest.raises(OutputColumnMismatchError):
        validate_and_align_output(
            df_missing, expected_missing, check_output_dtypes="raises"
        )

    # Test 9: Series with check_output_dtypes='warns' performs validation and casting
    result = validate_and_align_output(s, expected_s, check_output_dtypes="warns")
    assert result.dtype == np.dtype("float64")

    # Test 10: None expected_dtypes skips validation
    df_test = pd.DataFrame({"a": [1, 2, 3]})
    result = validate_and_align_output(df_test, None, check_output_dtypes="raises")
    pd.testing.assert_frame_equal(result, df_test)

    # Test 11: check_dtype_compatibility rejects float to int (causes truncation)
    with pytest.raises(OutputDtypeMismatchError) as exc_info:
        check_dtype_compatibility(np.dtype("float64"), np.dtype("int64"), "test_col")
    assert "float dtype" in str(exc_info.value).lower()
    assert "integer dtype" in str(exc_info.value).lower()
    assert "truncation" in str(exc_info.value).lower()

    # Test 12: check_dtype_compatibility allows int to float conversion
    result = check_dtype_compatibility(
        np.dtype("int64"), np.dtype("float64"), "test_col"
    )
    assert result is None

    # Test 13: validate_and_align_output warns on float to int with warns mode
    s_incompat = pd.Series([1.5, 2.5, 3.5], dtype=np.dtype("float64"))
    expected_incompat = np.dtype("int64")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = validate_and_align_output(
            s_incompat, expected_incompat, check_output_dtypes="warns"
        )
        assert len(w) > 0
        assert "truncation" in str(w[0].message).lower()

    # Test 14: validate_and_align_output rejects float to int with raises mode
    with pytest.raises(OutputDtypeMismatchError) as exc_info:
        validate_and_align_output(
            s_incompat, expected_incompat, check_output_dtypes="raises"
        )
    assert "truncation" in str(exc_info.value).lower()

    # Test 15: validate_and_align_output casts compatible types
    s_compat = pd.Series([1, 2, 3], dtype=np.dtype("int64"))
    expected_compat = np.dtype("float64")
    result = validate_and_align_output(
        s_compat, expected_compat, check_output_dtypes="warns"
    )
    assert result.dtype == np.dtype("float64")

    # Test 16: check_dtype_compatibility with blob types - matching types
    blob_dtype = ExternalBlobDtype()
    result = check_dtype_compatibility(blob_dtype, blob_dtype, "blob_col")
    assert result is None  # Same blob type should pass

    # Test 17: check_dtype_compatibility with Arrow blob type
    arrow_blob_dtype = ArrowDtype(ArrowBlobType())
    result = check_dtype_compatibility(arrow_blob_dtype, blob_dtype, "arrow_blob_col")
    assert result is None  # Arrow blob to blob should pass

    # Test 18: check_dtype_compatibility with blob to non-blob should fail
    with pytest.raises(OutputDtypeMismatchError):
        check_dtype_compatibility(blob_dtype, np.dtype("int64"), "blob_col")

    # Test 19: check_dtype_compatibility with non-blob to blob should fail
    with pytest.raises(OutputDtypeMismatchError):
        check_dtype_compatibility(np.dtype("int64"), blob_dtype, "int_col")

    # Test 20: mixed ArrowDtype vs numpy dtype (polars->pandas Arrow-backed)
    # int64[pyarrow] should be compatible with int64
    if is_arrow_dtype_supported():
        check_dtype_compatibility(
            ArrowDtype(pa.int64()), np.dtype("int64"), "arrow_int_col"
        )
        # double[pyarrow] should be compatible with float64
        check_dtype_compatibility(
            ArrowDtype(pa.float64()), np.dtype("float64"), "arrow_float_col"
        )
        # int32[pyarrow] should be compatible with int64 (safe widening)
        check_dtype_compatibility(
            ArrowDtype(pa.int32()), np.dtype("int64"), "arrow_widen_col"
        )
        # Reversed: numpy int64 vs ArrowDtype int64[pyarrow]
        check_dtype_compatibility(
            np.dtype("int64"), ArrowDtype(pa.int64()), "numpy_to_arrow_col"
        )
        # float64[pyarrow] to int64 should still fail (float->int truncation)
        with pytest.raises(OutputDtypeMismatchError):
            check_dtype_compatibility(
                ArrowDtype(pa.float64()), np.dtype("int64"), "float_to_int_col"
            )

    # Test 21: pandas ``object`` (plain-Python-string column) validates
    # against any declared string dtype and casts losslessly.  ``object`` is
    # the canonical dtype of a pandas string column, so rejecting it would
    # break every string-returning UDF validated with
    # check_output_dtypes="raises" (e.g. custom aggregations).
    declared_str_dtypes = (
        [(ArrowDtype(pa.string()), "string[pyarrow]")]
        if is_arrow_dtype_supported()
        else []
    ) + [
        (pd.StringDtype(), "string"),
        # pandas never materializes numpy-unicode (<U0) from an object
        # series — the cast is a no-op and object remains, which IS the
        # lossless numpy unicode equivalent — so validation passes and the
        # dtype assertion is "object".
        (np.dtype("<U0"), "object"),
    ]
    for expected, expected_dtype_str in declared_str_dtypes:
        check_dtype_compatibility(np.dtype("object"), expected, "object_to_str")
        result = validate_and_align_output(
            pd.DataFrame({"cols": ["a", "b"]}),
            pd.Series({"cols": expected}),
            check_output_dtypes="raises",
        )
        assert str(result["cols"].dtype) == expected_dtype_str
        assert list(result["cols"]) == ["a", "b"]
    # Non-string object content stays rejected under 'raises': the
    # object->string leniency must not become a blanket object pass — it
    # blesses only columns that genuinely hold Python strings, so dict/list/
    # mixed values raise instead of being silently stringified by the
    # subsequent astype.  Positive string-valued object columns still pass
    # (covered by the loop above).
    for bad_frame in (
        pd.DataFrame({"cols": [{"a": 1}]}),
        pd.DataFrame({"cols": [1, 2]}),
        pd.DataFrame({"cols": ["a", 1]}),
    ):
        with pytest.raises(OutputDtypeMismatchError):
            validate_and_align_output(
                bad_frame,
                pd.Series({"cols": pd.StringDtype()}),
                check_output_dtypes="raises",
            )
    # And a declared int still rejects an object column outright.
    with pytest.raises(OutputDtypeMismatchError):
        check_dtype_compatibility(np.dtype("object"), np.dtype("int64"), "obj_to_int")


@pytest.mark.skipif(
    ArrowVectorType is None,
    reason="need ArrowVectorType support to run this test",
)
@pytest.mark.parametrize(
    "element_type,dimension",
    [
        (pa.float32(), 128),
        (pa.float64(), 256),
    ],
)
def test_arrow_type_from_str_vector(element_type, dimension):
    vector_type = ArrowVectorType(element_type, dimension)
    type_str = str(vector_type)
    result = arrow_type_from_str(type_str)
    assert isinstance(result, ArrowVectorType)
    assert result.element_type == element_type
    assert result.dimension == dimension


@pytest.mark.parametrize("mode", ["warns", "raises"])
def test_validate_and_align_output_str_column_collision(mode):
    """Post-normalization label collisions (e.g. [0, "0"]) must raise in
    both warns and raises modes -- the frame is structurally ambiguous and
    cannot be aligned safely downstream."""
    df = pd.DataFrame([[1, 2]], columns=[0, "0"])
    expected = pd.Series({"0": np.dtype("int64")})
    with pytest.raises(OutputColumnMismatchError, match="Duplicate column labels"):
        validate_and_align_output(
            df,
            expected,
            check_output_dtypes=mode,
            normalize_str_columns=True,
        )


try:
    import polars as pl
except ImportError:
    pl = None


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_polars_string_result_declared_numpy_int_raises():
    """Regression: a polars String result declared
    ``np.dtype("int64")`` must raise ``OutputDtypeMismatchError``.  The
    native polars validator used to ``continue`` (skip) any column whose
    declared dtype was non-arrow (``_declared_dtype_to_arrow_type`` returns
    ``None`` for numpy/pandas dtypes), so the mismatch passed silently.
    """
    result = pl.DataFrame({"x": ["bad"]})
    declared = pd.Series({"x": np.dtype("int64")})
    with pytest.raises(OutputDtypeMismatchError):
        validate_and_align_output(result, declared, "raises", True)


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_polars_scalar_numpy_dtype_validates_not_crash():
    """Regression: a scalar ``np.dtype("int64")``
    (the helper accepts it as the dtype for every result column) must
    validate, not raise ``AttributeError`` on ``.index``.
    """
    result = pl.DataFrame({"x": ["bad"]})
    with pytest.raises(OutputDtypeMismatchError):
        validate_and_align_output(result, np.dtype("int64"), "raises", True)


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_polars_int64_result_declared_float64_casts():
    """Contract: a Polars Int64 result declared ``Float64`` casts natively."""
    result = pl.DataFrame({"x": [1, 2, 3]})
    declared = pd.Series({"x": np.dtype("float64")})
    out = validate_and_align_output(result, declared, "raises", True)
    assert out.schema["x"] == pl.Float64


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_polars_warns_extra_column_reorders_to_expected():
    """Contract: ``warns`` with columns ``["extra", "b", "a"]`` and
    declaration ``["a", "b"]`` warns and returns ``["a", "b"]`` (mirrors the
    pandas path).
    """
    result = pl.DataFrame({"extra": [1], "b": [2], "a": [3]})
    declared = pd.Series({"a": np.dtype("int64"), "b": np.dtype("int64")})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = validate_and_align_output(result, declared, "warns", True)
        assert len(w) > 0
        assert list(out.columns) == ["a", "b"]


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_polars_arrow_declared_dtype_still_validates():
    """Regression: an ArrowDtype-declared int64 column with a String result
    must still raise (the arrow path must not regress).
    """
    result = pl.DataFrame({"x": ["bad"]})
    declared = pd.Series({"x": wrap_arrow_dtype(pa.int64())})
    with pytest.raises(OutputDtypeMismatchError):
        validate_and_align_output(result, declared, "raises", True)


_INT64_REPRESENTATIONS = [
    pytest.param(np.dtype("int64"), id="numpy"),
    pytest.param(pl.Int64, id="polars"),
    pytest.param(pa.int64(), id="pyarrow"),
    pytest.param(wrap_arrow_dtype(pa.int64()), id="arrowdtype"),
]


@pytest.mark.skipif(pl is None, reason="polars not installed")
@pytest.mark.parametrize("declared", _INT64_REPRESENTATIONS)
def test_validate_polars_dtype_cast_representation_equivalent(declared):
    """Regression: all four representations of ``int64`` must behave
    identically.  The arrow-declared path used to raise because
    ``check_dtype_compatibility`` called a non-existent ``pa.types.can_cast``
    and the broad ``except`` conservatively rejected the cast; the numpy path
    succeeded via ``np.can_cast``.

    ``int32`` -> ``int64`` safely casts; ``float64`` -> ``int64`` raises
    (truncation guard preserved for every representation).
    """
    widening = pl.DataFrame({"x": pl.Series([1], dtype=pl.Int32)})
    out = validate_and_align_output(
        widening, pd.Series({"x": declared}), "raises", True
    )
    assert out.schema["x"] == pl.Int64

    truncation = pl.DataFrame({"x": pl.Series([1.5], dtype=pl.Float64)})
    with pytest.raises(OutputDtypeMismatchError):
        validate_and_align_output(
            truncation, pd.Series({"x": declared}), "raises", True
        )


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_polars_lossy_decimal_cast_rejected():
    """Regression: a ``pl.Decimal(10, 2)`` result declared
    ``decimal128(10, 1)`` must raise ``OutputDtypeMismatchError`` instead of
    silently rounding values (``[1.25, 9.99]`` became ``[1.2, 10.0]``).
    Decimal precision/scale has no faithful NumPy representation, so only an
    exact parameter match is compatible.
    """
    from decimal import Decimal

    result = pl.DataFrame(
        {"x": pl.Series([Decimal("1.25"), Decimal("9.99")], dtype=pl.Decimal(10, 2))}
    )
    declared = pd.Series({"x": wrap_arrow_dtype(pa.decimal128(10, 1))})
    with pytest.raises(OutputDtypeMismatchError):
        validate_and_align_output(result, declared, "raises", True)


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_polars_decimal_to_object_lossless():
    """Regression: a Decimal result declared ``object`` is a lossless
    conversion and must fall through the normal compatibility path instead
    of being rejected by the Decimal guard (which only fires when BOTH
    sides are Decimal with unequal parameters).
    """
    from decimal import Decimal

    result = pl.DataFrame({"x": pl.Series([Decimal("1.25")], dtype=pl.Decimal(10, 2))})
    declared = pd.Series({"x": np.dtype("O")})
    out = validate_and_align_output(result, declared, "raises", True)
    # Polars keeps the Decimal column (object has no polars equivalent to
    # cast to); the guard must not reject, and the value is preserved.
    assert out["x"][0] == Decimal("1.25")


@pytest.mark.skipif(pl is None, reason="polars not installed")
@pytest.mark.parametrize("mode", ["warns", "raises"])
def test_validate_polars_duplicate_label_after_normalization_rejected(mode):
    """Regression: a declared Series indexed ``[0, "0"]`` collapses to one
    ``"0"`` entry after string normalization; it must raise the existing
    duplicate-label ``OutputColumnMismatchError`` (both modes, mirroring the
    pandas path) instead of silently dropping one declaration.
    """
    result = pl.DataFrame({"0": [1]})
    declared = pd.Series([np.dtype("int64"), np.dtype("float64")], index=[0, "0"])
    with pytest.raises(OutputColumnMismatchError):
        validate_and_align_output(result, declared, mode, True)


@pytest.mark.skipif(pl is None, reason="polars not installed")
@pytest.mark.parametrize("mode", ["warns", "raises"])
def test_validate_polars_value_dependent_cast_failure(mode):
    """Regression: an Int64 value ``200`` declared ``int8`` overflows at cast
    time; the polars ``InvalidOperationError`` must not leak.  ``raises``
    produces ``OutputDtypeMismatchError``; ``warns`` emits the configured
    warning and keeps the original column dtype.
    """
    result = pl.DataFrame({"x": pl.Series([200], dtype=pl.Int64)})
    declared = pd.Series({"x": np.dtype("int8")})
    if mode == "raises":
        with pytest.raises(OutputDtypeMismatchError):
            validate_and_align_output(result, declared, mode, True)
    else:
        with pytest.warns(FutureWarning):
            out = validate_and_align_output(result, declared, mode, True)
        assert out.schema["x"] == pl.Int64


@pytest.mark.parametrize("mode", ["ignore", "warns", "raises"])
def test_validate_pandas_canonicalizes_int_label_before_policy(mode):
    """Regression: a result with integer label 0 must be canonicalized to
    "0" before the dtype-policy early exit, so even "ignore" returns a
    string-labeled frame (LiteFrame boundary contract).
    """
    result = pd.DataFrame({0: [1, 2, 3]})
    expected = pd.Series({"0": np.dtype("int64")})
    out = validate_and_align_output(result, expected, mode, normalize_str_columns=True)
    assert list(out.columns) == ["0"]


@pytest.mark.parametrize("mode", ["ignore", "warns", "raises"])
def test_validate_pandas_duplicate_label_after_normalization_rejected(mode):
    """Regression: a result containing both integer 0 and string "0" must be
    rejected as a duplicate-column collision after string normalization,
    in every policy (the collision guard runs before the early exit).
    """
    result = pd.DataFrame({0: [1], "0": [2]})
    expected = pd.Series({"0": np.dtype("int64")})
    with pytest.raises(OutputColumnMismatchError, match="Duplicate column labels"):
        validate_and_align_output(result, expected, mode, normalize_str_columns=True)
