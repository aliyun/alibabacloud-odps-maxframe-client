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

from typing import Any, Callable, Dict, List, MutableMapping, Tuple, Union

import numpy as np
import pandas as pd

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.dataframe.core import (
    DATAFRAME_GROUPBY_TYPE,
    GROUPBY_TYPE,
    DataFrameGroupBy,
    IndexValue,
    SeriesGroupBy,
)
from maxframe.dataframe.groupby.utils import (
    warn_axis_argument,
    warn_prepend_index_group_keys,
)
from maxframe.dataframe.operators import DataFrameOperator, DataFrameOperatorMixin
from maxframe.dataframe.type_infer import (
    InferredDataFrameMeta,
    infer_dataframe_return_value,
    prepend_group_keys_as_index,
)
from maxframe.dataframe.utils import (
    copy_func_scheduling_hints,
    make_column_list,
    parse_index,
    validate_output_types,
)
from maxframe.serialization.serializables import (
    DictField,
    FieldTypes,
    FunctionField,
    Int32Field,
    ListField,
    TupleField,
)
from maxframe.udf import BuiltinFunction, MarkedFunction
from maxframe.utils import (
    copy_if_possible,
    deprecate_positional_args,
    make_dtype,
    make_dtypes,
    pd_release_version,
)
from maxframe.utils.functional import check_closure_for_entities

_apply_without_group_keys = pd_release_version < (1, 5, 0)
_has_include_groups = (2, 2, 0) <= pd_release_version < (3, 0, 0)


class GroupByApplyChunk(DataFrameOperatorMixin, DataFrameOperator):
    _op_type_ = opcodes.APPLY_CHUNK
    _op_module_ = "dataframe.groupby"

    func = FunctionField("func")
    batch_rows = Int32Field("batch_rows", default=None)
    args = TupleField("args", default=None)
    kwargs = DictField("kwargs", default=None)

    groupby_params = DictField("groupby_params", default=None)
    order_cols = ListField("order_cols", default=None)
    ascending = ListField("ascending", FieldTypes.bool, default_factory=lambda: [True])

    def __init__(self, output_type=None, **kw):
        if output_type:
            kw["_output_types"] = [output_type]
        super().__init__(**kw)
        if hasattr(self, "func"):
            copy_func_scheduling_hints(self.func, self)

    def has_custom_code(self) -> bool:
        return not isinstance(self.func, BuiltinFunction)

    def _call_dataframe(self, df, dtypes, dtype, name, index_value, element_wise):
        # return dataframe
        if self.output_types[0] == OutputType.dataframe:
            dtypes = make_dtypes(dtypes)
            # apply_chunk will use generate new range index for results
            return self.new_dataframe(
                [df],
                shape=df.shape if element_wise else (np.nan, len(dtypes)),
                index_value=index_value,
                columns_value=parse_index(dtypes.index, store_data=True),
                dtypes=dtypes,
            )

        # return series
        return self.new_series(
            [df], shape=(np.nan,), name=name, dtype=dtype, index_value=index_value
        )

    def _call_series(self, series, dtypes, dtype, name, index_value, element_wise):
        if self.output_types[0] == OutputType.series:
            shape = series.shape if element_wise else (np.nan,)
            return self.new_series(
                [series],
                dtype=dtype,
                shape=shape,
                index_value=index_value,
                name=name,
            )

        dtypes = make_dtypes(dtypes)
        return self.new_dataframe(
            [series],
            shape=(np.nan, len(dtypes)),
            index_value=index_value,
            columns_value=parse_index(dtypes.index, store_data=True),
            dtypes=dtypes,
        )

    def __call__(
        self,
        groupby: Union[DataFrameGroupBy, SeriesGroupBy],
        dtypes: Union[Tuple[str, Any], Dict[str, Any]] = None,
        dtype: Any = None,
        name: Any = None,
        output_type=None,
        index=None,
        skip_infer: bool = False,
        prepend_index_group_keys: bool = True,
    ):
        input_df = groupby.inputs[0]
        if isinstance(input_df, GROUPBY_TYPE):
            input_df = input_df.inputs[0]

        # if skip_infer, directly build a frame
        if self.output_types and self.output_types[0] == OutputType.df_or_series:
            return self.new_df_or_series([input_df])

        # infer return index and dtypes
        inferred_meta = self._infer_batch_func_returns(
            groupby,
            output_type=output_type,
            dtypes=dtypes,
            dtype=dtype,
            name=name,
            index=index,
            skip_infer=skip_infer,
            prepend_index_group_keys=prepend_index_group_keys,
        )

        if inferred_meta.index_value is None:
            inferred_meta.index_value = parse_index(
                None, (groupby.key, groupby.index_value.key, self.func)
            )
        inferred_meta.check_absence("output_type", "dtypes", "dtype")

        if isinstance(groupby, DATAFRAME_GROUPBY_TYPE):
            return self._call_dataframe(
                input_df,
                dtypes=inferred_meta.dtypes,
                dtype=inferred_meta.dtype,
                name=inferred_meta.name,
                index_value=inferred_meta.index_value,
                element_wise=inferred_meta.elementwise,
            )

        return self._call_series(
            input_df,
            dtypes=inferred_meta.dtypes,
            dtype=inferred_meta.dtype,
            name=inferred_meta.name,
            index_value=inferred_meta.index_value,
            element_wise=inferred_meta.elementwise,
        )

    def _infer_batch_func_returns(
        self,
        input_groupby: Union[DataFrameGroupBy, SeriesGroupBy],
        output_type: OutputType,
        dtypes: Union[pd.Series, List[Any], Dict[str, Any]] = None,
        dtype: Any = None,
        name: Any = None,
        index: Union[pd.Index, IndexValue] = None,
        elementwise: bool = None,
        skip_infer: bool = False,
        prepend_index_group_keys: bool = True,
    ) -> InferredDataFrameMeta:
        def infer_func(groupby_obj):
            args = copy_if_possible(self.args or ())
            kwargs = copy_if_possible(self.kwargs or {})

            in_obj = input_groupby
            while isinstance(in_obj, GROUPBY_TYPE):
                in_obj = in_obj.inputs[0]

            by_cols = (
                make_column_list(self.groupby_params.get("by"), in_obj.dtypes) or []
            )
            if not self.groupby_params.get("selection"):
                selection = [
                    c for c in input_groupby.inputs[0].dtypes.index if c not in by_cols
                ]
                groupby_obj = groupby_obj[selection]
            if not _has_include_groups:
                kwargs.pop("include_groups", None)
            res = groupby_obj.apply(self.func, *args, **kwargs)
            if _apply_without_group_keys and not (
                prepend_index_group_keys
                or res.index.names != groupby_obj.obj.index.names
            ):
                # Need to patch group_index for legacy local pandas version
                #  only when index names not changed
                # FIXME here we add `not prepend_index_group_keys` to solely make
                #  our behavior consistent with legacy implementations. It should
                #  be removed once the argument is dropped
                res.index = prepend_group_keys_as_index(res.index, input_groupby)
            return res

        # Set __wrapped__ so unwrap_function(infer_func) reaches self.func,
        infer_func.__wrapped__ = self.func

        inferred_meta = infer_dataframe_return_value(
            input_groupby,
            infer_func,
            output_type=output_type,
            dtypes=dtypes,
            dtype=dtype,
            name=name,
            index=index,
            elementwise=elementwise,
            skip_infer=skip_infer,
            prepend_index_group_keys=prepend_index_group_keys,
        )

        # merge specified and inferred index, dtypes, output_type
        # elementwise used to decide shape
        self.output_types = (
            [inferred_meta.output_type]
            if not self.output_types and inferred_meta.output_type
            else self.output_types
        )
        if self.output_types:
            inferred_meta.output_type = self.output_types[0]
        inferred_meta.dtypes = dtypes if dtypes is not None else inferred_meta.dtypes
        inferred_meta.elementwise = elementwise or inferred_meta.elementwise
        return inferred_meta

    @classmethod
    def estimate_size(
        cls,
        ctx: MutableMapping[str, Union[int, float]],
        op: "GroupByApplyChunk",
    ) -> None:
        if isinstance(op.func, MarkedFunction):
            ctx[op.outputs[0].key] = float("inf")
        super().estimate_size(ctx, op)


@deprecate_positional_args
def df_groupby_apply_chunk(
    dataframe_groupby,
    func: Union[str, Callable],
    batch_rows=None,
    *,
    dtypes=None,
    dtype=None,
    name=None,
    output_type=None,
    index=None,
    skip_infer=False,
    order_cols=None,
    ascending=True,
    prepend_index_group_keys=False,
    check_output_dtypes=None,
    args=(),
    **kwargs,
):
    """
    Apply function `func` group-wise and combine the results together.
    The pandas DataFrame given to the function is a chunk of the input
    dataframe, consider as a batch rows.

    The function passed to `apply` must take a dataframe as its first
    argument and return a DataFrame, Series or scalar. `apply` will
    then take care of combining the results back together into a single
    dataframe or series. `apply` is therefore a highly flexible
    grouping method.

    Don't expect to receive all rows of the DataFrame in the function,
    as it depends on the implementation of MaxFrame and the internal
    running state of MaxCompute.

    Parameters
    ----------
    func : callable
        A callable that takes a dataframe as its first argument, and
        returns a dataframe, a series or a scalar. In addition, the
        callable may take positional and keyword arguments.

    batch_rows : int
        Specify expected number of rows in a batch, as well as the len of
        function input dataframe. When the remaining data is insufficient,
        it may be less than this number.

    output_type : {'dataframe', 'series'}, default None
        Specify type of returned object. See `Notes` for more details.

    dtypes : Series, default None
        Specify dtypes of returned DataFrames. See `Notes` for more details.

    dtype : numpy.dtype, default None
        Specify dtype of returned Series. See `Notes` for more details.

    name : str, default None
        Specify name of returned Series. See `Notes` for more details.

    index : Index, default None
        Specify index of returned object. See `Notes` for more details.

    skip_infer: bool, default False
        Whether to skip inferring dtypes when dtypes or output_type is not
        specified. Once specified as True, you need to explicitly specify
        dtypes and output_type via arguments or type annotations of
        the function.

    prepend_index_group_keys: bool, default False
        If True, the index of returned dataframe or series will automatically
        contain group keys if ``as_index=True``, or group indexes if
        ``as_index=False``, when ``group_keys=True``. It will also exclude
        group keys in user function inputs by default. See notes for more
        details.

        .. note::

            ``prepend_index_group_keys`` will be set to True by default in
            future releases, and a warning will be shown if the parameter
            is set to False. To make sure your code works in future
            releases, please set this to True and remove group indexes
            in index parameter or type annotation of ``func``.

    check_output_dtypes : {'ignore', 'warns', 'raises'}, default None
        Validation mode for output dtypes and columns. When specified,
        validates that the user function returns data with expected dtypes.

        - 'ignore': No validation performed
        - 'warns': Validate and show warnings on mismatch (default when None)
        - 'raises': Validate and raise errors on mismatch

        Note: Group columns are automatically excluded from validation as they
        are managed separately by the groupby infrastructure.

    args, kwargs : tuple and dict
        Optional positional and keyword arguments to pass to ``func``.

    Returns
    -------
    applied : Series or DataFrame

    See Also
    --------
    Series.apply : Apply a function to a Series.
    DataFrame.apply : Apply a function to each row or column of a DataFrame.
    DataFrame.mf.apply_chunk : Apply a function to row batches of a DataFrame.

    Notes
    -----
    When deciding output dtypes and shape of the return value, MaxFrame will
    try applying ``func`` onto a mock grouped object, and the apply call
    may fail. When this happens, you need to specify the type of apply
    call (DataFrame or Series) in output_type.

    * For DataFrame output, you need to specify a list or a pandas Series
      as ``dtypes`` of output DataFrame.
    * For Series output, you need to specify ``dtype`` and ``name`` of
      output Series.
    * ``index`` determines index of output DataFrame or Series. You may specify
      a dummy pandas index indicating the names and types of index of the output
      of ``func``, for instance, ``pd.MultiIndex.from_tuples([("a", 0)], names=["key1", "key2"])``.
      If ``index`` is not supplied, index of the input DataFrame or Series will
      be used. When `prepend_index_group_keys` is True, the index of the returning
      object will be ``index`` prepended with group information given ``as_index``
      and ``group_keys`` argument of the ``groupby`` function, which is consistent
      with pandas 3.0. When ``prepend_index_group_keys`` is False, you must specify
      a mock index with all fields, including group keys. As it is complicated to
      pass full index definition, ``prepend_index_group_keys=False`` will be
      deprecated in near future. Please supply ``prepend_index_group_keys=True``
      where possible.

    MaxFrame adopts expected behavior of pandas>=3.0 by ignoring group columns
    in user function input. If you still need a group column for your function
    input, try selecting it right after `groupby` results, for instance,
    ``df.groupby("A")[["A", "B", "C"]].mf.apply_chunk(func)`` will pass data of
    column A into ``func``.

    The ``batch_rows`` parameter controls memory usage. Larger values may improve
    performance but increase OOM risk. Ensure sufficient worker memory when
    processing entire groups in a single batch.

    Examples
    --------
    Example 1: Filter rows within each group
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Find employees with salary above a threshold in each department.
    This demonstrates how the result index shows intra-group positions (0-n).

    >>> import maxframe.dataframe as md
    >>> import pandas as pd
    >>>
    >>> # Create sample employee data
    >>> data = {
    ...     'department': ['HR', 'HR', 'HR', 'IT', 'IT', 'IT', 'Finance', 'Finance'],
    ...     'employee_id': [1, 2, 3, 4, 5, 6, 7, 8],
    ...     'salary': [50000, 55000, 60000, 70000, 75000, 80000, 90000, 95000],
    ...     'years_experience': [2, 3, 5, 1, 4, 6, 3, 7]
    ... }
    >>> df = md.DataFrame(data)
    >>> df.execute()
      department  employee_id  salary  years_experience
    0         HR            1   50000                 2
    1         HR            2   55000                 3
    2         HR            3   60000                 5
    3         IT            4   70000                 1
    4         IT            5   75000                 4
    5         IT            6   80000                 6
    6    Finance            7   90000                 3
    7    Finance            8   95000                 7

    >>> def filter_high_salary(batch_df):
    ...     # batch_df contains employee data for a single department
    ...     # Group key (department) is NOT included in the DataFrame columns
    ...     print(f"Processing {len(batch_df)} rows, received {batch_df}", flush=True)
    ...
    ...     # Filter: keep employees with salary > 55000
    ...     return batch_df[batch_df['salary'] > 55000]
    >>>
    >>> # Specify dtypes without the group key column (department)
    >>> result_dtypes = df.dtypes[['employee_id', 'salary', 'years_experience']]
    >>>
    >>> result = df.groupby('department').mf.apply_chunk(
    ...     filter_high_salary,
    ...     output_type='dataframe',
    ...     dtypes=result_dtypes,
    ...     prepend_index_group_keys=True,
    ... )
    >>> result.execute()
                  employee_id  salary  years_experience
    department
    Finance    6            7   90000                 3
               7            8   95000                 7
    HR         2            3   60000                 5
    IT         3            4   70000                 1
               4            5   75000                 4
               5            6   80000                 6

    Result explanation:
    - The first level index ("department") shows the group key values
    - The second level index (2, 3, 4, 5, 6, 7...) are the ORIGINAL row indices from the input DataFrame
    - For Finance department: employees at original indices 6-7 meet the criteria
    - For HR department: employee at original index 2 meets the criteria
    - For IT department: employees at original indices 3-5 meet the criteria
    - Group keys are NOT included in the batch_df in the UDF input by default, but are included in the result
    - When specifying dtypes, exclude group key columns (they are indexes in the result)

    Example 2: Return DataFrame with single aggregation column
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Apply a function to calculate average salary by department,
    returning a DataFrame with a single column and explicit type specifications.
    This example introduces the ``batch_rows`` parameter to control batch size.

    >>> # Specify dtypes with type annotations
    >>> def calculate_avg_salary(batch_df) -> pd.DataFrame['avg_salary': 'float64']:
    ...     # Important: batch_df contains only non-group columns by default
    ...     # Group keys are not included in the UDF input
    ...     print(f"Processing batch with {len(batch_df)} rows")
    ...
    ...     # Return a single value as DataFrame - internal index is preserved by design
    ...     avg_val = batch_df['salary'].mean()
    ...     return pd.DataFrame({'avg_salary': [avg_val]})
    >>>
    >>> result = df.groupby('department').mf.apply_chunk(
    ...     calculate_avg_salary,
    ...     batch_rows=2,  # Process 2 rows per batch
    ...     prepend_index_group_keys=True,
    ... )
    >>> result.execute()
                  avg_salary
    department
    Finance    0     92500.0
    HR         0     52500.0
               0     60000.0
    IT         0     72500.0
               0     80000.0

    Result explanation:
    - The first level index ("department") shows the group key values
    - The second level index ('0') is newly created because each UDF call returns a single-row DataFrame
    - HR department shows two rows because batch_rows=2 caused two separate UDF calls
    - Finance and IT departments were processed in single batches
    - When UDF returns aggregated results, the index is from newly created dataframe

    Example 3: Including group keys in UDF input
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Sometimes you need access to group keys within your UDF. This example
    shows how to include them by explicitly selecting the group column
    along with other columns. We'll filter high-salary employees but this
    time include the department column in the UDF input.

    >>> def filter_high_salary_with_dept(batch_df) -> pd.DataFrame[
    ...     'department': 'object', 'employee_id': 'int64', 'salary': 'float64'
    ... ]:
    ...     # Now batch_df includes the department column since we explicitly selected it
    ...     department = batch_df['department'].iloc[0]
    ...     print(f"Processing {len(batch_df)} rows for department: {department}")
    ...
    ...     # Filter: keep employees with salary > 55000 (same logic as Example 1)
    ...     return batch_df[batch_df['salary'] > 55000]
    >>>
    >>> # Include the group key by explicitly selecting it with other columns
    >>> result = df.groupby('department')[['department', 'employee_id', 'salary']].mf.apply_chunk(
    ...     filter_high_salary_with_dept, prepend_index_group_keys=True
    ... )
    >>> result.execute()
                 department  employee_id   salary
    department
    Finance    6    Finance            7  90000.0
               7    Finance            8  95000.0
    HR         2         HR            3  60000.0
    IT         3         IT            4  70000.0
               4         IT            5  75000.0
               5         IT            6  80000.0

    Result explanation:
    - The first level index ("department") shows the group key values
    - The second level index (2, 3, 4, 5, 6, 7...) are the ORIGINAL row indices from the input DataFrame
    - By selecting ['department', 'employee_id', 'salary'], we ensure the department column is available in UDF
    - The UDF can now access department values (though not used in this simple filter)
    - Original indices are preserved in the result
    - The filter logic is the same as Example 1: salary > 55000

    This example demonstrates how to explicitly include group keys in your UDF
    by selecting them in the groupby operation, making them available for use
    within your function if needed.

    Example 4: Explicitly specifying output types and index
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    When UDFs cannot be executed locally for inference, you must explicitly
    specify output_type, dtypes, and index via arguments or type annotation
    to ensure correct execution.

    >>> def create_summary_stats(batch_df):
    ...     # Calculate basic statistics
    ...     avg_salary = batch_df['salary'].mean()
    ...     total_salary = batch_df['salary'].sum()
    ...     employee_count = len(batch_df)
    ...
    ...     # Return DataFrame with correct types
    ...     result_df = pd.DataFrame({
    ...         'avg_salary': pd.Series([avg_salary], dtype='float64'),
    ...         'total_salary': pd.Series([total_salary], dtype='float64'),
    ...         'employee_count': pd.Series([employee_count], dtype='int64')
    ...     })
    ...
    ...     return result_df
    >>>
    >>> # Create inner index returned by UDF
    >>> result_index = pd.Index([], dtype='int64', name='inner_index')
    >>>
    >>> # Explicitly specify all output parameters
    >>> result = df.groupby('department').mf.apply_chunk(
    ...     create_summary_stats,
    ...     batch_rows=10000,
    ...     output_type='dataframe',  # specifies output type as DataFrame
    ...     dtypes={
    ...         'avg_salary': 'float64',
    ...         'total_salary': 'float64',
    ...         'employee_count': 'int'
    ...     }, # specifies the final dataframe column types
    ...     index=result_index,  # specifies the structure of the final MultiIndex result
    ...     prepend_index_group_keys=True,
    ... )
    >>> result.execute()
                           avg_salary  total_salary  employee_count
    department inner_index
    Finance    0            92500.0      185000.0                2
    HR         0            55000.0      165000.0                3
    IT         0            75000.0      225000.0                3

    Result explanation:
    - The first level index ("department") shows the group key values (string type)
    - The second level index ("inner_index") comes from the UDF's returned DataFrame (int type)
    - output_type='dataframe' tells MaxFrame to expect DataFrame output
    - dtypes defines exact column types to prevent inference errors
    - index parameter specifies the structure of the final MultiIndex result
    - batch_rows=10000 ensures entire groups are processed together

    To simplify output type definition, you can also use type annotations.
    In the code snippet below, pd.DataFrame shows the returning type is
    a DataFrame with index names 'inner_index' and columns 'avg_salary',
    'total_salary', 'employee_count'. Types of both indexes and columns are
    also specified.

    >>> def create_summary_stats(batch_df) -> pd.DataFrame[
    ...     {'inner_index': 'int64'},  # type of index
    ...     {'avg_salary': 'float64', 'total_salary': 'float64', 'employee_count': 'int64'},  # type of data
    ... ]:
    ...     # details of function omitted

    Key takeaway: Always specify output_type and dtypes when:
    1. UDF creates new DataFrame structures
    2. Local inference might fail
    3. You need consistent output format

    Note: The index parameter defines the inner index structure when
    prepend_index_group_keys=True is specified, and the resulting index
    combines group keys (first level, string) and UDF indices (second level, int).
    """
    if not prepend_index_group_keys:
        warn_prepend_index_group_keys(dataframe_groupby)
    else:
        kwargs = kwargs.copy()
        kwargs["include_groups"] = False
    warn_axis_argument("mf.apply_chunk", kwargs)

    if not isinstance(func, Callable):
        raise TypeError("function must be a callable object")

    if batch_rows is not None:
        if not isinstance(batch_rows, int):
            raise TypeError("batch_rows must be an integer")
        elif batch_rows <= 0:
            raise ValueError("batch_rows must be greater than 0")

    if dtype is not None:
        dtype = make_dtype(dtype)

    output_types = kwargs.pop("output_types", None)
    object_type = kwargs.pop("object_type", None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type
    )
    output_type = output_types[0] if output_types else None
    if skip_infer and output_type is None:
        output_type = (
            OutputType.dataframe if dataframe_groupby.ndim == 2 else OutputType.series
        )

    if order_cols and not isinstance(order_cols, list):
        order_cols = [order_cols]
    if not isinstance(ascending, list):
        ascending = [ascending]
    elif len(order_cols) != len(ascending):
        raise ValueError("order_cols and ascending must have same length")

    # Check for entities captured in closure
    check_closure_for_entities(func, operation_name="groupby_apply_chunk")

    # bind args and kwargs
    op = GroupByApplyChunk(
        func=func,
        batch_rows=batch_rows,
        output_type=output_type,
        args=args,
        kwargs=kwargs,
        order_cols=order_cols,
        ascending=ascending,
        groupby_params=(dataframe_groupby.op.groupby_params or {}).copy(),
    )

    # Store check_output_dtypes in extra_params if specified
    if check_output_dtypes is not None:
        if not hasattr(op, "extra_params") or op.extra_params is None:
            op.extra_params = {}
        op.extra_params["check_output_dtypes"] = check_output_dtypes

    return op(
        dataframe_groupby,
        dtypes=dtypes,
        dtype=dtype,
        name=name,
        index=index,
        output_type=output_type,
        prepend_index_group_keys=prepend_index_group_keys,
    )
