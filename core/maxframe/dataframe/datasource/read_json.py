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

from typing import MutableMapping, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from ...protocol import DefaultIndexType

try:
    from pyarrow import NativeFile
except ImportError:  # pragma: no cover
    NativeFile = None

from ... import opcodes
from ...config import options
from ...core import OutputType
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    Int32Field,
    Int64Field,
    StringField,
)
from ...utils import lazy_import, no_default
from ..utils import (
    parse_index,
    to_arrow_dtypes,
    validate_default_index_type,
    validate_dtype_backend,
)
from .core import (
    ColumnPruneSupportedDataSourceMixin,
    DtypeBackendCompatibleMixin,
    LakeDataSource,
)
from .utils import get_lake_output_info, iter_local_files

cudf = lazy_import("cudf")


class DataFrameReadJSON(
    LakeDataSource,
    ColumnPruneSupportedDataSourceMixin,
    DtypeBackendCompatibleMixin,
):
    _op_type_ = opcodes.READ_JSON

    orient = StringField("orient")
    typ = StringField("typ")
    dtype = AnyField("dtype")
    convert_axes = BoolField("convert_axes")
    lines = BoolField("lines")
    chunksize = Int64Field("chunksize")
    compression = StringField("compression")
    index_col = Int32Field("index_col")
    usecols = AnyField("usecols")
    keep_usecols_order = BoolField("keep_usecols_order", default=None)
    chunk_bytes = StringField("chunk_bytes", default=None)
    read_kwargs = DictField("read_kwargs", default=None)
    head_bytes = StringField("head_bytes", default=None)
    head_lines = Int64Field("head_lines", default=None)

    def __init__(self, output_type=None, **kwargs):
        if output_type is not None:
            kwargs["_output_types"] = [output_type]
        super().__init__(**kwargs)

    def get_columns(self):
        return self.usecols

    def set_pruned_columns(self, columns, *, keep_order=None):
        self.usecols = columns
        self.keep_usecols_order = keep_order

    def __call__(self, chunk_bytes=None, **kwargs):
        if self.read_stage is not None:
            # output for planning or meta fetching
            self._output_types = [OutputType.scalar]
            return self.new_tileable(None, shape=(), dtype=np.dtype("O"))

        shape = (
            (np.nan, len(kwargs["dtypes"]))
            if self.output_types[0] == OutputType.dataframe
            else (np.nan,)
        )
        return self.new_tileable(None, shape=shape, chunk_bytes=chunk_bytes, **kwargs)

    @classmethod
    def estimate_size(
        cls, ctx: MutableMapping[str, Union[int, float]], op: "DataFrameReadJSON"
    ):  # pragma: no cover
        # todo implement this to facilitate local computation
        ctx[op.outputs[0].key] = float("inf")


def read_json(
    path,
    *,
    orient=None,
    typ="frame",
    dtype=None,
    convert_axes=None,
    lines=False,
    chunksize=None,
    compression="infer",
    index_col=None,
    usecols=None,
    chunk_bytes="64M",
    gpu=None,
    head_bytes="100k",
    head_lines=None,
    default_index_type: Union[DefaultIndexType, str] = None,
    use_nullable_dtypes: bool = no_default,
    dtype_backend: str = no_default,
    storage_options: dict = None,
    memory_scale: int = None,
    merge_small_files: bool = True,
    merge_small_file_options: dict = None,
    session=None,
    run_kwargs: dict = None,
    **kwargs,
):
    r"""
    Read a JSON file into a DataFrame.

    Parameters
    ----------
    path : str, path object, or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be: file://localhost/path/to/table.json,
        you can also read from external resources using a URL like:
        hdfs://localhost:8020/test.json.
        If you want to pass in a path object, pandas accepts any ``os.PathLike``.
        By file-like object, we refer to objects with a ``read()`` method, such as
        a file handler (e.g. via builtin ``open`` function) or ``StringIO``.
    orient : str, optional
        Indication of expected JSON string format.
        Compatible JSON strings can be produced by ``to_json()`` with a
        corresponding orient value.
        The set of possible orients is:
        - ``'split'`` : dict like ``{'index' -> [index], 'columns' -> [columns], 'data' -> [values]}``
        - ``'records'`` : list like ``[{column -> value}, ... , {column -> value}]``
        - ``'index'`` : dict like ``{index -> {column -> value}}``
        - ``'columns'`` : dict like ``{column -> {index -> value}}``
        - ``'values'`` : just the values array
        The allowed and default values depend on the value of the `typ` parameter.
        * when ``typ == 'series'``,
          - allowed orients are ``{'split','records','index'}``
          - default is ``'index'``
          - The Series index must be unique for orient ``'index'``.
        * when ``typ == 'frame'``,
          - allowed orients are ``{'split','records','index','columns','values'}``
          - default is ``'columns'``
          - The DataFrame index must be unique for orients ``'index'`` and ``'columns'``.
          - The DataFrame columns must be unique for orients ``'index'``, ``'columns'``,
            and ``'records'``.
    typ : {{'frame', 'series'}}, default 'frame'
        The type of object to recover.
    dtype : bool or dict, default None
        If True, infer dtypes; if a dict of column to dtype, then use those;
        if False, then don't infer dtypes at all, applies only to the data.
    convert_axes : bool, default None
        Try to convert the axes to the proper dtypes.
    convert_dates : bool or list of str, default True
        List of columns to parse for dates. If True, then try to parse datelike columns.
        A column label is datelike if
        * it ends with ``'_at'``,
        * it ends with ``'_time'``,
        * it begins with ``'date'``, or
        * it is ``'datetime'``, ``'timestamp'``, ``'modified'``, or ``'created'``.
    keep_default_dates : bool, default True
        If parsing dates, then parse the default datelike columns.
    precise_float : bool, default False
        Set to enable usage of higher precision (strtod) function when
        decoding string to double values. Default (False) is to use fast but
        less precise builtin functionality.
    date_unit : str, default None
        The timestamp unit to detect if converting dates. The default behaviour
        is to try and detect the correct precision, but if this is not desired
        then pass one of 's', 'ms', 'us' or 'ns' to force parsing only seconds,
        milliseconds, microseconds or nanoseconds respectively.
    encoding : str, default is 'utf-8'
        The encoding to use to decode py3 bytes.
    lines : bool, default False
        Read the file as a json object per line.
    chunksize : int, optional
        Return JsonReader object for iteration.
        See the `IO Tools docs
        <https://pandas.pydata.org/pandas-docs/stable/io.html#io-jsonl>`_
        for more information on ``chunksize``.
        This can only be passed if `lines=True`.
        If this is None, the file will be read into memory all at once.
    compression : {{'infer', 'gzip', 'bz2', 'zip', 'xz', None}}, default 'infer'
        For on-the-fly decompression of on-disk data. If 'infer' and
        `filepath_or_buffer` is path-like, then detect compression from the
        following extensions: '.gz', '.bz2', '.zip', or '.xz' (otherwise no
        decompression). If using 'zip', the ZIP file must contain only one data
        file to be read in. Set to None for no decompression.
    index_col : int, str, sequence of int / str, or False, default ``None``
      Column(s) to use as the row labels of the ``DataFrame``, either given as
      string name or column index. If a sequence of int / str is given, a
      MultiIndex is used.
      Note: ``index_col=False`` can be used to force pandas to *not* use the first
      column as the index, e.g. when you have a malformed file with delimiters at
      the end of each line.
    usecols : list-like or callable, optional
        Return a subset of the columns. If list-like, all elements must either
        be positional (i.e. integer indices into the document columns) or strings
        that correspond to column names provided either by the user in `names` or
        inferred from the document header row(s). For example, a valid list-like
        `usecols` parameter would be ``[0, 1, 2]`` or ``['foo', 'bar', 'baz']``.
        Element order is ignored, so ``usecols=[0, 1]`` is the same as ``[1, 0]``.
        To instantiate a DataFrame from ``data`` with element order preserved use
        ``pd.read_csv(data, usecols=['foo', 'bar'])[['foo', 'bar']]`` for columns
        in ``['foo', 'bar']`` order or
        ``pd.read_csv(data, usecols=['foo', 'bar'])[['bar', 'foo']]``
        for ``['bar', 'foo']`` order.
        If callable, the callable function will be evaluated against the column
        names, returning names where the callable function evaluates to True. An
        example of a valid callable argument would be ``lambda x: x.upper() in
        ['AAA', 'BBB', 'DDD']``. Using this parameter results in much faster
        parsing time and lower memory usage.
    chunk_bytes: int, float or str, optional
        Number of chunk bytes.
    gpu: bool, default False
        If read into cudf DataFrame.
    head_bytes: int, float or str, optional
        Number of bytes to use in the head of file, mainly for data inference.
    head_lines: int, optional
        Number of lines to use in the head of file, mainly for data inference.
    default_index_type: {None, 'range', 'incremental'}, default None
        If index_col not specified, specify type of index to generate.
        If not specified, `options.dataframe.default_index_type` will be used.
    dtype_backend: {'numpy', 'pyarrow'}, default 'numpy'
        Back-end data type applied to the resultant DataFrame (still experimental).
    storage_options: dict, optional
        Options for storage connection.
    merge_small_files: bool, default True
        Merge small files whose size is small.
    merge_small_file_options: dict
        Options for merging small files

    Returns
    -------
    DataFrame or Series
        A JSON file is returned as two-dimensional data structure with labeled axes.

    See Also
    --------
    to_json : Convert DataFrame to JSON string.
    json_normalize : Normalize semi-structured JSON data into a flat table.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> md.read_json('data.json')  # doctest: +SKIP
    >>> # read from HDFS
    >>> md.read_json('hdfs://localhost:8020/test.json')  # doctest: +SKIP
    >>> # read from OSS
    >>> md.read_json('oss://oss-cn-hangzhou.aliyuncs.com/bucket/test.json',
    >>>             storage_options={'role_arn': 'acs:ram::xxxxxx:role/aliyunodpsdefaultrole'})
    """
    from .dataframe import from_pandas as from_pandas_df
    from .series import from_pandas as from_pandas_series

    default_index_type = validate_default_index_type(default_index_type, **kwargs)
    local_test_mode = kwargs.pop("_local_test_mode", False)
    single_path = path[0] if isinstance(path, list) else path
    parsed_path = urlparse(single_path)
    if not local_test_mode and (
        not parsed_path.scheme or parsed_path.scheme.lower() == "file"
    ):
        # just read locally when path is not remote
        local_dfs = []
        for path, part_keys in iter_local_files(path):
            kw = {}
            if use_nullable_dtypes is not no_default:
                kw = {"use_nullable_dtypes": use_nullable_dtypes}
            if dtype_backend is not no_default:
                kw = {"dtype_backend": dtype_backend}
            sub_df = pd.read_json(
                path,
                orient=orient,
                typ=typ,
                dtype=dtype,
                convert_axes=convert_axes,
                lines=lines,
                chunksize=chunksize,
                compression=compression,
                **kw,
            )
            for k, v in part_keys or ():
                sub_df[k] = v
            local_dfs.append(sub_df)
        df = pd.concat(local_dfs) if len(local_dfs) > 1 else local_dfs[0]
        return from_pandas_df(df) if typ == "frame" else from_pandas_series(df)

    common_kwargs = dict(
        orient=orient,
        typ=typ,
        convert_axes=convert_axes,
        lines=lines,
        chunksize=chunksize,
        compression=compression,
        index_col=index_col,
        usecols=usecols,
        use_nullable_dtypes=use_nullable_dtypes,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
        read_kwargs=kwargs,
    )
    # Get dtypes, index_dtypes and index_value using the common utility function
    result = get_lake_output_info(
        DataFrameReadJSON,
        path=path,
        dtype=dtype,
        head_bytes=head_bytes,
        head_lines=head_lines,
        default_index_type=default_index_type,
        session=session,
        run_kwargs=run_kwargs,
        **common_kwargs,
    )

    dtypes = result.dtypes
    index_dtypes = result.index_dtypes
    index_value = result.index_value
    is_partitioned = result.is_partitioned
    output_type = result.output_type

    # Handle series case
    ser_dtype = name = None
    if output_type == OutputType.series:
        ser_dtype = dtypes.iloc[0] if len(dtypes) > 0 else None
        name = dtypes.index[0] if len(dtypes.index) > 0 else None

    # For JSON, we need to combine index_dtypes with dtypes for the full_dtypes
    full_dtypes = (
        pd.concat([index_dtypes, dtypes]) if index_dtypes is not None else dtypes
    )
    default_index_type = None if index_dtypes is not None else default_index_type

    if output_type == OutputType.series:
        columns_value = None
    else:
        columns_value = parse_index(dtypes.index, store_data=True)

    chunk_bytes = chunk_bytes or options.chunk_store_limit
    op = DataFrameReadJSON(
        path=path,
        dtype=full_dtypes,
        gpu=gpu,
        default_index_type=default_index_type,
        is_partitioned=is_partitioned,
        memory_scale=memory_scale,
        merge_small_files=merge_small_files,
        merge_small_file_options=merge_small_file_options,
        chunk_bytes=chunk_bytes,
        output_type=output_type,
        **common_kwargs,
    )
    dtype_backend = validate_dtype_backend(
        dtype_backend or options.dataframe.dtype_backend
    )

    if not gpu and dtype_backend == "pyarrow":
        dtypes = to_arrow_dtypes(dtypes)
    if output_type == OutputType.dataframe:
        ret = op(index_value=index_value, columns_value=columns_value, dtypes=dtypes)
    else:
        ret = op(index_value=index_value, name=name, dtype=ser_dtype)
    return ret
