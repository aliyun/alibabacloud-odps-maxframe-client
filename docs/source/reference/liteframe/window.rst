.. _generated.liteframe_window:

Window functions
================

Rolling, expanding, and exponentially weighted (EW) window objects are returned by
:meth:`maxframe.liteframe.LiteFrame.rolling`,
:meth:`maxframe.liteframe.LiteFrame.expanding`, and
:meth:`maxframe.liteframe.LiteFrame.ewm` respectively.
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

.. currentmodule:: maxframe.liteframe.window.ewm

EWM methods
-----------
.. autosummary::
   :toctree: generated/

   LiteFrameEWM
   LiteFrameEWM.agg
   LiteFrameEWM.mean
   LiteFrameEWM.std
   LiteFrameEWM.var
