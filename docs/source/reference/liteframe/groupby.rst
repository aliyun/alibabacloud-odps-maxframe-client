.. _generated.liteframe_groupby:

GroupBy
=======
.. currentmodule:: maxframe.liteframe.groupby

GroupBy objects are returned by :meth:`maxframe.liteframe.LiteFrame.groupby`.
Aggregation output is always column-based (equivalent to ``as_index=False``)
with no MultiIndex — group keys are returned as regular columns.

Function application
--------------------
.. autosummary::
   :toctree: generated/

   LiteFrameGroupBy.apply
   LiteFrameGroupBy.flat_apply
   LiteFrameGroupBy.apply_chunk
   LiteFrameGroupBy.transform
   LiteFrameGroupBy.transform_chunk
   LiteFrameGroupBy.agg

Computations / descriptive stats
--------------------------------
.. autosummary::
   :toctree: generated/

   LiteFrameGroupBy.all
   LiteFrameGroupBy.any
   LiteFrameGroupBy.count
   LiteFrameGroupBy.cumcount
   LiteFrameGroupBy.cummax
   LiteFrameGroupBy.cummin
   LiteFrameGroupBy.cumprod
   LiteFrameGroupBy.cumsum
   LiteFrameGroupBy.max
   LiteFrameGroupBy.mean
   LiteFrameGroupBy.median
   LiteFrameGroupBy.min
   LiteFrameGroupBy.nunique
   LiteFrameGroupBy.prod
   LiteFrameGroupBy.rolling
   LiteFrameGroupBy.expanding
   LiteFrameGroupBy.sem
   LiteFrameGroupBy.size
   LiteFrameGroupBy.skew
   LiteFrameGroupBy.kurt
   LiteFrameGroupBy.std
   LiteFrameGroupBy.sum
   LiteFrameGroupBy.var

Window transforms
-----------------
.. autosummary::
   :toctree: generated/

   LiteFrameGroupBy.shift
   LiteFrameGroupBy.diff
   LiteFrameGroupBy.pct_change
   LiteFrameGroupBy.ffill
   LiteFrameGroupBy.bfill
