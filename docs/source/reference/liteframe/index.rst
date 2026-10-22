.. _liteframe_api:

MaxFrame LiteFrame
==================

This page gives an overview of all public MaxFrame LiteFrame objects, functions and
methods. All classes and functions exposed in ``maxframe.liteframe.*`` namespace are public.

LiteFrame is a performance-dedicated tabular data type in MaxFrame with **no index
semantics**, designed for simplified column-based operations with Arrow dtypes as
first-class citizens. It provides a pandas-looking DataFrame API that eliminates the
complexity of index management — no implicit label alignment, no unexpected frame
stacking during aggregation, and no silent type promotion when integer columns contain
nulls. By removing index overhead and embedding element-wise expressions in a small
set of fusible operators, LiteFrame achieves lower planning cost and higher execution
throughput.

LiteFrame is **not** a replacement for the existing MaxFrame DataFrame. It serves a
complementary role: a simpler, index-free substrate for workflows that do not need full
pandas Index behavior. Both LiteFrame and DataFrame coexist in MaxFrame — choose LiteFrame
for straightforward column-oriented data processing, or use DataFrame when you need
index-aware operations.

All operations are lazy: they return LiteFrame objects that are executed on demand via
``.execute()``. Row order is preserved through execution unless an operator explicitly
reorders rows, enabling positional APIs like ``iloc``, ``head``, ``shift``, and
cumulative operations.

.. note::

   The LiteFrame API is currently in beta. While functional, it may evolve
   in future releases — method names, parameters, and behavior may change
   based on feedback and further development.

.. toctree::
   :maxdepth: 2

   io
   general_functions
   frame
   groupby
   accessors
   window
