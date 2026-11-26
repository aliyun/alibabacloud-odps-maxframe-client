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

from typing import List

import pandas as pd

from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.codegen.spe.liteframe.expressions import _materialize_range_columns
from maxframe.liteframe.operators.sort import LiteFrameSort
from maxframe.liteframe.utils import next_unselected


@register_op_adapter(LiteFrameSort)
class LiteFrameSortAdapter(SPEOperatorAdapter):
    @staticmethod
    def _run_sort(
        df: pd.DataFrame,
        by,
        ascending,
        na_position: str,
        nrows,
        keep_kind: str,
        range_columns: dict = None,
    ) -> pd.DataFrame:
        # Materialize range columns if present (sort changes row order)
        df = _materialize_range_columns(df, range_columns)

        if nrows is None or keep_kind == "head":
            result = df.sort_values(by=by, ascending=ascending, na_position=na_position)
            if nrows is not None:
                result = result.head(nrows)
            return result

        if keep_kind in ("first", "last"):
            if isinstance(ascending, bool):
                is_ascending = ascending
                is_descending = not ascending
            else:
                is_ascending = all(ascending)
                is_descending = not any(ascending)
            if is_ascending or is_descending:
                # pandas < 2.0 drops missing keys from nsmallest/nlargest,
                # while pandas >= 2.0 keeps them to fill up to ``nrows`` when
                # there are fewer non-null keys. Normalize by working on a
                # positional index (so rows can be identified unambiguously
                # even when the frame index has duplicate labels) and padding
                # the missing-key rows back in sort order.
                orig_index = df.index
                pos_df = df.reset_index(drop=True)
                if is_ascending:
                    result = pos_df.nsmallest(nrows, columns=by, keep=keep_kind)
                else:
                    result = pos_df.nlargest(nrows, columns=by, keep=keep_kind)
                ordered_df = pos_df.sort_values(
                    by=by, ascending=ascending, na_position=na_position, kind="stable"
                )
                result = LiteFrameSortAdapter._pad_short_selection(
                    ordered_df, result, nrows
                )
                result.index = orig_index.take(result.index.to_numpy())
            else:
                # Mixed sort directions: nsmallest/nlargest cannot express
                # per-column directions, so fall back to sort + head.
                result = df.sort_values(
                    by=by, ascending=ascending, na_position=na_position
                ).head(nrows)
            return result.sort_values(
                by=by,
                ascending=ascending,
                na_position=na_position,
                kind="stable",
            )

        # keep_kind == "all"
        # Use kind='stable' to ensure tied rows are in original order,
        # matching pandas keep='all' behavior.
        sorted_df = df.sort_values(
            by=by, ascending=ascending, na_position=na_position, kind="stable"
        )
        return LiteFrameSortAdapter._include_all_ties(
            sorted_df, by, ascending, na_position, nrows
        )

    @staticmethod
    def _pad_short_selection(
        ordered_df: pd.DataFrame,
        result: pd.DataFrame,
        nrows,
    ) -> pd.DataFrame:
        """Fill a short selection with the next rows in sort order.

        pandas can return fewer rows than ``min(nrows, len(ordered_df))``: it
        drops rows with missing sort keys (pandas < 2.0) and, with Arrow-backed
        keys, returns an empty selection whenever a key column holds a missing
        value. ``ordered_df`` is the input in sort order carrying a positional
        (``RangeIndex``) index, so the rows the sort places first among the
        unselected ones fill the gap, complete-key rows included, and the padded
        rows follow the key order of the selection. Rows are addressed by their
        original labels, which are not the sorted frame's positions.
        """
        target = min(nrows, len(ordered_df))
        if len(result) >= target:
            return result

        pad_labels = next_unselected(
            ordered_df.index.to_numpy(), result.index.to_numpy(), target - len(result)
        )
        if pad_labels.size == 0:
            return result
        return pd.concat([result, ordered_df.loc[pad_labels]])

    @staticmethod
    def _include_all_ties(
        sorted_df: pd.DataFrame,
        by,
        ascending,
        na_position: str,
        nrows: int,
    ) -> pd.DataFrame:
        """Include all rows that tie with the boundary row in the top-n result."""
        if nrows == 0:
            return sorted_df.head(0)
        if len(sorted_df) <= nrows:
            return sorted_df

        top_n = sorted_df.head(nrows)
        boundary_values = top_n[by].iloc[-1]

        # Build mask for rows matching the boundary sort key values
        mask = pd.Series(True, index=sorted_df.index)
        for col in by:
            bval = boundary_values[col]
            if pd.isna(bval):
                mask &= sorted_df[col].isna()
            else:
                mask &= sorted_df[col] == bval

        # All boundary-matching rows from the full sorted DataFrame
        all_at_boundary = sorted_df[mask]
        # Rows in top_n that are strictly above the boundary.
        # Use positional access (iloc) to avoid label-expansion bugs
        # when the input index contains duplicate labels.
        top_n_above = top_n.iloc[~mask.iloc[:nrows].values]
        # Combine: rows above boundary + all tied rows at boundary
        return pd.concat([top_n_above, all_at_boundary])

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        res_var = context.get_output_tileable_variable(op.outputs[0])

        # Extract range_columns from input tileable's frame_metadata
        range_columns = None
        in_tileable = op.inputs[0]
        if hasattr(in_tileable, "frame_metadata") and in_tileable.frame_metadata:
            if in_tileable.frame_metadata.range_columns:
                range_columns = {}
                for name, info in in_tileable.frame_metadata.range_columns.items():
                    range_columns[name] = (info.start, info.stop, info.step)

        args = self._translate_call_args(
            context,
            op.by,
            op.ascending,
            op.na_position,
            op.nrows,
            op.keep_kind,
            range_columns=range_columns,
        )

        return [f"{res_var} = {cls_name}._run_sort({input_var}, {', '.join(args)})"]
