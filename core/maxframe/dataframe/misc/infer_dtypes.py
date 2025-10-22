# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

from ... import opcodes
from ...serialization.serializables import AnyField, StringField
from ..core import DATAFRAME_TYPE, SERIES_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin


class DataFrameInferDtypes(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_INFER_DTYPES

    infer_method = StringField("infer_method")
    infer_kwargs = AnyField("infer_kwargs")

    infer_stage = StringField("infer_stage", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def __call__(self, df):
        if isinstance(df, DATAFRAME_TYPE):
            return self.new_dataframe(
                [df],
                shape=df.shape,
                dtypes=None,
                index_value=df.index_value,
                columns_value=df.columns_value,
            )
        else:
            assert isinstance(df, SERIES_TYPE)
            return self.new_series(
                [df],
                shape=df.shape,
                dtype=None,
                name=df.name,
                index_value=df.index_value,
            )


def convert_dtypes(
    df_or_series,
    infer_objects=True,
    convert_string=True,
    convert_integer=True,
    convert_boolean=True,
    convert_floating=True,
    dtype_backend="numpy",
):
    """
    Convert columns to best possible dtypes using dtypes supporting ``pd.NA``.

    Parameters
    ----------
    infer_objects : bool, default True
        Whether object dtypes should be converted to the best possible types.
    convert_string : bool, default True
        Whether object dtypes should be converted to ``StringDtype()``.
    convert_integer : bool, default True
        Whether, if possible, conversion can be done to integer extension types.
    convert_boolean : bool, defaults True
        Whether object dtypes should be converted to ``BooleanDtypes()``.
    convert_floating : bool, defaults True
        Whether, if possible, conversion can be done to floating extension types.
        If `convert_integer` is also True, preference will be give to integer
        dtypes if the floats can be faithfully casted to integers.

    Returns
    -------
    Series or DataFrame
        Copy of input object with new dtype.

    See Also
    --------
    infer_objects : Infer dtypes of objects.
    to_datetime : Convert argument to datetime.
    to_timedelta : Convert argument to timedelta.
    to_numeric : Convert argument to a numeric type.

    Notes
    -----
    By default, ``convert_dtypes`` will attempt to convert a Series (or each
    Series in a DataFrame) to dtypes that support ``pd.NA``. By using the options
    ``convert_string``, ``convert_integer``, ``convert_boolean`` and
    ``convert_boolean``, it is possible to turn off individual conversions
    to ``StringDtype``, the integer extension types, ``BooleanDtype``
    or floating extension types, respectively.

    For object-dtyped columns, if ``infer_objects`` is ``True``, use the inference
    rules as during normal Series/DataFrame construction.  Then, if possible,
    convert to ``StringDtype``, ``BooleanDtype`` or an appropriate integer
    or floating extension type, otherwise leave as ``object``.

    If the dtype is integer, convert to an appropriate integer extension type.

    If the dtype is numeric, and consists of all integers, convert to an
    appropriate integer extension type. Otherwise, convert to an
    appropriate floating extension type.

    .. versionchanged:: 1.2
        Starting with pandas 1.2, this method also converts float columns
        to the nullable floating extension type.

    In the future, as new dtypes are added that support ``pd.NA``, the results
    of this method will change to support those new dtypes.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame(
    ...     {
    ...         "a": md.Series([1, 2, 3], dtype=mt.dtype("int32")),
    ...         "b": md.Series(["x", "y", "z"], dtype=mt.dtype("O")),
    ...         "c": md.Series([True, False, mt.nan], dtype=mt.dtype("O")),
    ...         "d": md.Series(["h", "i", mt.nan], dtype=mt.dtype("O")),
    ...         "e": md.Series([10, mt.nan, 20], dtype=mt.dtype("float")),
    ...         "f": md.Series([mt.nan, 100.5, 200], dtype=mt.dtype("float")),
    ...     }
    ... )

    Start with a DataFrame with default dtypes.

    >>> df.execute()
       a  b      c    d     e      f
    0  1  x   True    h  10.0    NaN
    1  2  y  False    i   NaN  100.5
    2  3  z    NaN  NaN  20.0  200.0

    >>> df.dtypes.execute()
    a      int32
    b     object
    c     object
    d     object
    e    float64
    f    float64
    dtype: object

    Convert the DataFrame to use best possible dtypes.

    >>> dfn = df.convert_dtypes()
    >>> dfn.execute()
       a  b      c     d     e      f
    0  1  x   True     h    10   <NA>
    1  2  y  False     i  <NA>  100.5
    2  3  z   <NA>  <NA>    20  200.0

    >>> dfn.dtypes.execute()
    a      Int32
    b     string
    c    boolean
    d     string
    e      Int64
    f    Float64
    dtype: object

    Start with a Series of strings and missing data represented by ``np.nan``.

    >>> s = md.Series(["a", "b", mt.nan])
    >>> s.execute()
    0      a
    1      b
    2    NaN
    dtype: object

    Obtain a Series with dtype ``StringDtype``.

    >>> s.convert_dtypes().execute()
    0       a
    1       b
    2    <NA>
    dtype: string
    """
    dtype_backend = "numpy" if dtype_backend == "numpy_nullable" else dtype_backend
    op = DataFrameInferDtypes(
        infer_method="convert_dtypes",
        infer_kwargs=dict(
            infer_objects=infer_objects,
            convert_string=convert_string,
            convert_integer=convert_integer,
            convert_boolean=convert_boolean,
            convert_floating=convert_floating,
            dtype_backend=dtype_backend,
        ),
    )
    return op(df_or_series)


def infer_objects(df_or_series, copy=True):
    """
    Attempt to infer better dtypes for object columns.

    Attempts soft conversion of object-dtyped
    columns, leaving non-object and unconvertible
    columns unchanged. The inference rules are the
    same as during normal Series/DataFrame construction.

    Returns
    -------
    converted : same type as input object

    See Also
    --------
    to_datetime : Convert argument to datetime.
    to_timedelta : Convert argument to timedelta.
    to_numeric : Convert argument to numeric type.
    convert_dtypes : Convert argument to best possible dtype.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({"A": ["a", 1, 2, 3]})
    >>> df = df.iloc[1:]
    >>> df.execute()
       A
    1  1
    2  2
    3  3

    >>> df.dtypes.execute()
    A    object
    dtype: object

    >>> df.infer_objects().dtypes.execute()
    A    int64
    dtype: object
    """
    if (isinstance(df_or_series, SERIES_TYPE) and df_or_series.dtype != "O") or (
        isinstance(df_or_series, DATAFRAME_TYPE)
        and all(dt != "O" for dt in df_or_series.dtypes)
    ):
        # no objects to cast
        return df_or_series

    _ = copy  # in MaxFrame data are immutable, thus ignore the parameter
    op = DataFrameInferDtypes(
        infer_method="infer_objects",
        infer_kwargs={},
    )
    return op(df_or_series)
