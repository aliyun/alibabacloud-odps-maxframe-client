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

import copy
from typing import List, Optional

import pandas as pd

try:
    import polars as _pl
except ImportError:  # pragma: no cover
    _pl = None

from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.liteframe.operators.reduction import LiteFrameAgg
from maxframe.liteframe.reduction.custom_agg import CustomAgg
from maxframe.udf import BuiltinFunction
from maxframe.utils import validate_and_align_output


def _resolve_custom_func(custom_func):
    """Resolve a custom_func to the actual callable (BuiltinFunction -> .func)."""
    if isinstance(custom_func, BuiltinFunction):
        return custom_func.func
    return custom_func


def _to_input_df(df, input_frame_type):
    """Convert a pandas frame to the requested UDF input frame type.

    SPE stores intermediate frames as pandas; a polars-mode custom agg
    receives a polars DataFrame, matching the DPE contract.
    """
    if input_frame_type == "polars" and _pl is not None:
        return _pl.from_pandas(df)
    return df


def _from_polars_result(result):
    """Normalize a custom-agg result back to pandas (SPE's storage form)."""
    if _pl is not None and isinstance(result, _pl.DataFrame):
        return result.to_pandas(use_pyarrow_extension_array=True)
    if _pl is not None and isinstance(result, _pl.Series):
        return result.to_frame().to_pandas(use_pyarrow_extension_array=True)
    return result


def _call_custom_agg(actual_func, input_df, agg_output_dtypes=None):
    """Call the agg function on a single frame, handling class/instance/plain.

    Mirrors ``LiteFrameAggExecutor._call_custom_agg`` in DPE: a
    ``CustomAgg`` subclass is instantiated, an instance is deep-copied per
    call (so state/resources don't cross group boundaries), and ``close()``
    is released in ``finally``.  A plain callable is invoked directly.
    """
    if isinstance(actual_func, type) and issubclass(actual_func, CustomAgg):
        instance = actual_func()
    elif isinstance(actual_func, CustomAgg):
        instance = copy.deepcopy(actual_func)
    else:
        result = actual_func(input_df)
        return _validate_agg_output(result, agg_output_dtypes)
    try:
        result = instance.agg(input_df)
    finally:
        if hasattr(instance, "close"):
            instance.close()
    return _validate_agg_output(result, agg_output_dtypes)


def _validate_agg_output(result, agg_output_dtypes):
    """Validate a custom-agg result against declared output dtypes."""
    if agg_output_dtypes is None or len(agg_output_dtypes) == 0:
        return result
    return validate_and_align_output(
        result,
        agg_output_dtypes,
        check_output_dtypes="raises",
        normalize_str_columns=True,
    )


@register_op_adapter(LiteFrameAgg)
class LiteFrameAggAdapter(SPEOperatorAdapter):
    @staticmethod
    def _run_custom_whole_frame_agg(
        df: pd.DataFrame,
        custom_func,
        input_frame_type: str,
        agg_output_dtypes=None,
    ) -> pd.DataFrame:
        """Apply a custom agg to the whole (non-grouped) frame."""
        actual_func = _resolve_custom_func(custom_func)
        input_df = _to_input_df(df, input_frame_type)
        result = _call_custom_agg(actual_func, input_df, agg_output_dtypes)
        result = _from_polars_result(result)
        if isinstance(result, pd.Series):
            result = result.to_frame().T
        return result

    @staticmethod
    def _run_custom_groupby_agg(
        df: pd.DataFrame,
        custom_func,
        groupby_params: dict,
        input_frame_type: str,
        agg_output_dtypes=None,
        agg_cols=None,
    ) -> pd.DataFrame:
        """Apply a custom agg per group.

        ``agg_cols`` narrows each group's input to the columns the
        compilation declared (selection-narrowed, group-key-excluded),
        so unselected columns do not leak into the user agg.  Group keys
        are re-materialized afterward from the group tuple.
        """
        by = groupby_params["by"]
        by_cols = [by] if isinstance(by, str) else list(by)
        dropna = groupby_params.get("dropna", True)
        sort = groupby_params.get("sort", True)

        actual_func = _resolve_custom_func(custom_func)
        grouped = df.groupby(by_cols, dropna=dropna, sort=sort)

        results = []
        for key, group_df in grouped:
            # Narrow to declared columns (group-key-excluded).  An explicit
            # empty list (``groupby("key")[[]]``) feeds a zero-column frame
            # with the group's original height — matching the groupby
            # selection contract.
            if agg_cols is None:
                input_cols = [c for c in group_df.columns if c not in by_cols]
            else:
                input_cols = [c for c in agg_cols if c in group_df.columns]
            group_input = group_df[input_cols] if input_cols else group_df.iloc[:, :0]
            group_input = _to_input_df(group_input, input_frame_type)

            res = _call_custom_agg(actual_func, group_input, agg_output_dtypes)
            res = _from_polars_result(res)

            if not isinstance(res, pd.DataFrame):
                res = pd.DataFrame([res])
            # Materialize group keys so the user agg need not echo them.
            # Prepend (insert at front) to honor the declared key-first
            # column order from compile_custom_agg; assigning ``res[{col}]``
            # would append the key column *after* the UDF result columns.
            if len(by_cols) == 1:
                res.insert(0, by_cols[0], key)
            else:
                for i, (k, col) in enumerate(zip(key, by_cols)):
                    res.insert(i, col, k)
            results.append(res)

        if not results:
            # Zero groups: empty input, or group keys all null with
            # dropna=True.  Mirror the DPE/groupby contract: return a
            # zero-row frame with the declared schema (source key dtypes
            # plus the declared agg output dtypes) instead of raising on
            # ``pd.concat([])``.
            declared = {col: df[col].dtype for col in by_cols}
            if agg_output_dtypes is not None:
                declared.update(agg_output_dtypes.to_dict())
            return pd.DataFrame(
                {col: pd.Series(dtype=dtype) for col, dtype in declared.items()}
            )

        result = pd.concat(results, ignore_index=True)
        # Group keys must retain the source key dtype: scalar assignment into
        # the UDF frame re-infers an Arrow-backed string key as ``object``.
        for col in by_cols:
            result[col] = result[col].astype(df[col].dtype)
        return result

    @staticmethod
    def _run_custom_agg(
        df: pd.DataFrame,
        agg_funcs: list,
        groupby_params: Optional[dict],
    ) -> pd.DataFrame:
        """Execute a custom agg (single-chunk, associative)."""
        desc = next(d for d in agg_funcs if d.get("custom_func"))
        custom_func = desc["custom_func"]
        input_frame_type = desc.get("input_frame_type", "pandas")
        agg_output_dtypes = desc.get("agg_output_dtypes")
        agg_cols = desc.get("cols")

        if groupby_params:
            return LiteFrameAggAdapter._run_custom_groupby_agg(
                df,
                custom_func,
                groupby_params,
                input_frame_type,
                agg_output_dtypes,
                agg_cols,
            )
        return LiteFrameAggAdapter._run_custom_whole_frame_agg(
            df, custom_func, input_frame_type, agg_output_dtypes
        )

    @staticmethod
    def _run_agg(
        df: pd.DataFrame,
        agg_funcs: list,
        groupby_params: Optional[dict],
        output_column_names: Optional[list],
    ) -> pd.DataFrame:
        # Custom aggregation: dispatch to the custom-agg path, which calls
        # the user's CustomAgg.agg per group / whole frame and validates the
        # output schema — mirroring DPE's LiteFrameAggExecutor.
        if any(d.get("custom_func") for d in agg_funcs):
            return LiteFrameAggAdapter._run_custom_agg(df, agg_funcs, groupby_params)

        use_prefix = len(agg_funcs) > 1

        if groupby_params:
            by = groupby_params["by"]
            by_cols = [by] if isinstance(by, str) else list(by)
            dropna = groupby_params.get("dropna", True)
            sort = groupby_params.get("sort", True)

            grouped = df.groupby(by_cols, sort=sort, dropna=dropna)

            result_series = []
            column_names = []

            for desc in agg_funcs:
                func_name = desc["map_func_name"]
                cols = desc["cols"]
                prefix = f"{desc['output_key']}__" if use_prefix else ""
                pre_agg_col = desc.get("pre_agg_col")
                kwds = desc.get("kwds") or {}

                if pre_agg_col is not None:
                    s = grouped[pre_agg_col].sum()
                    for c in cols:
                        col_name = f"{prefix}{c}_{func_name}" if prefix else c
                        column_names.append(col_name)
                        result_series.append(s)
                elif func_name == "size":
                    col_name = f"{prefix}__size__" if prefix else "__size__"
                    column_names.append(col_name)
                    result_series.append(grouped.size())
                elif func_name == "nunique":
                    dropna_kw = kwds.get("dropna", True)
                    for c in cols:
                        col_name = f"{prefix}{c}_{func_name}" if prefix else c
                        column_names.append(col_name)
                        result_series.append(grouped[c].nunique(dropna=dropna_kw))
                else:
                    for c in cols:
                        col_name = f"{prefix}{c}_{func_name}" if prefix else c
                        column_names.append(col_name)
                        result_series.append(getattr(grouped[c], func_name)())

            result = pd.concat(result_series, axis=1)
            result.columns = column_names
            result = result.reset_index()
        else:
            result_data = {}
            for desc in agg_funcs:
                func_name = desc["map_func_name"]
                cols = desc["cols"]
                prefix = f"{desc['output_key']}__" if use_prefix else ""
                pre_agg_col = desc.get("pre_agg_col")
                kwds = desc.get("kwds") or {}

                if pre_agg_col is not None:
                    for c in cols:
                        col_name = f"{prefix}{c}_{func_name}" if prefix else c
                        result_data[col_name] = [df[pre_agg_col].sum()]
                elif func_name == "size":
                    col_name = f"{prefix}__size__" if prefix else "__size__"
                    result_data[col_name] = [len(df)]
                elif func_name == "nunique":
                    dropna_kw = kwds.get("dropna", True)
                    for c in cols:
                        col_name = f"{prefix}{c}_{func_name}" if prefix else c
                        result_data[col_name] = [df[c].nunique(dropna=dropna_kw)]
                else:
                    for c in cols:
                        col_name = f"{prefix}{c}_{func_name}" if prefix else c
                        result_data[col_name] = [getattr(df[c], func_name)()]

            result = pd.DataFrame(result_data)

        if output_column_names:
            current_cols = list(result.columns)
            if len(current_cols) != len(output_column_names):
                raise ValueError(
                    f"Column count mismatch: result has {len(current_cols)} "
                    f"columns {current_cols}, expected "
                    f"{len(output_column_names)} columns {output_column_names}"
                )
            rename_map = {
                old: new
                for old, new in zip(current_cols, output_column_names)
                if old != new
            }
            if rename_map:
                result = result.rename(columns=rename_map)

        return result

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        res_var = context.get_output_tileable_variable(op.outputs[0])

        args = self._translate_call_args(
            context,
            op.agg_funcs,
            op.groupby_params,
            op.output_column_names,
        )
        return [f"{res_var} = {cls_name}._run_agg({input_var}, {', '.join(args)})"]
