.. _generated.groupby:

GroupBy
=======
.. currentmodule:: maxframe.dataframe.groupby

GroupBy objects are returned by groupby
calls: :func:`maxframe.dataframe.DataFrame.groupby`, :func:`maxframe.dataframe.Series.groupby`, etc.

Indexing, iteration
-------------------
.. autosummary::
   :toctree: generated/

.. currentmodule:: maxframe.dataframe.groupby

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_without_autosummary.rst

.. currentmodule:: maxframe.dataframe.groupby

Function application
--------------------
.. autosummary::
   :toctree: generated/

   GroupBy.agg
   GroupBy.aggregate

Computations / descriptive stats
--------------------------------
.. autosummary::
   :toctree: generated/

   GroupBy.all
   GroupBy.any
   GroupBy.count
   GroupBy.max
   GroupBy.mean
   GroupBy.median
   GroupBy.min
   GroupBy.size
   GroupBy.sem
   GroupBy.std
   GroupBy.sum
   GroupBy.var

The following methods are available in both ``SeriesGroupBy`` and
``DataFrameGroupBy`` objects, but may differ slightly, usually in that
the ``DataFrameGroupBy`` version usually permits the specification of an
axis argument, and often an argument indicating whether to restrict
application to columns of a specific data type.

.. autosummary::
   :toctree: generated/

   DataFrameGroupBy.count
   DataFrameGroupBy.nunique

The following methods are available only for ``SeriesGroupBy`` objects.

.. autosummary::
   :toctree: generated/


The following methods are available only for ``DataFrameGroupBy`` objects.

.. autosummary::
   :toctree: generated/
