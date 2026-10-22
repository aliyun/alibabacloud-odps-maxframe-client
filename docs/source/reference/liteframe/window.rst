.. _generated.liteframe_window:

Window functions
================

Rolling and expanding window objects are returned by
:meth:`maxframe.liteframe.LiteFrame.rolling` and
:meth:`maxframe.liteframe.LiteFrame.expanding` respectively.
Call an aggregation method to produce a LiteFrame.

.. currentmodule:: maxframe.liteframe.window.rolling

Rolling methods
---------------
.. autosummary::
   :toctree: generated/

   LiteFrameRolling
   LiteFrameRolling.agg
   LiteFrameRolling.sum
   LiteFrameRolling.mean
   LiteFrameRolling.min
   LiteFrameRolling.max
   LiteFrameRolling.std
   LiteFrameRolling.var
   LiteFrameRolling.median
   LiteFrameRolling.count

.. currentmodule:: maxframe.liteframe.window.expanding

Expanding methods
-----------------
.. autosummary::
   :toctree: generated/

   LiteFrameExpanding
   LiteFrameExpanding.agg
   LiteFrameExpanding.sum
   LiteFrameExpanding.mean
   LiteFrameExpanding.min
   LiteFrameExpanding.max
   LiteFrameExpanding.std
   LiteFrameExpanding.var
   LiteFrameExpanding.count
