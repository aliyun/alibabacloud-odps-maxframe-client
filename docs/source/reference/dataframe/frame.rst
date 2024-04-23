.. _generated.dataframe:

DataFrame
=========
.. currentmodule:: maxframe.dataframe

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Axes**

.. autosummary::
   :toctree: generated/

   DataFrame.index
   DataFrame.columns

.. autosummary::
   :toctree: generated/

   DataFrame.dtypes
   DataFrame.select_dtypes
   DataFrame.ndim
   DataFrame.shape

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.astype
   DataFrame.copy
   DataFrame.isna
   DataFrame.notna

Indexing, iteration
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.head
   DataFrame.insert
   DataFrame.pop
   DataFrame.query

Binary operator functions
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.add
   DataFrame.sub
   DataFrame.mul
   DataFrame.div
   DataFrame.truediv
   DataFrame.floordiv
   DataFrame.mod
   DataFrame.pow
   DataFrame.dot
   DataFrame.radd
   DataFrame.rsub
   DataFrame.rmul
   DataFrame.rdiv
   DataFrame.rtruediv
   DataFrame.rfloordiv
   DataFrame.rmod
   DataFrame.rpow
   DataFrame.lt
   DataFrame.gt
   DataFrame.le
   DataFrame.ge
   DataFrame.ne
   DataFrame.eq

Function application, GroupBy & window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.apply
   DataFrame.agg
   DataFrame.aggregate
   DataFrame.groupby
   DataFrame.transform

.. _generated.dataframe.stats:

Computations / descriptive stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.abs
   DataFrame.all
   DataFrame.any
   DataFrame.count
   DataFrame.describe
   DataFrame.max
   DataFrame.mean
   DataFrame.min
   DataFrame.nunique
   DataFrame.pct_change
   DataFrame.prod
   DataFrame.product
   DataFrame.quantile
   DataFrame.round
   DataFrame.sem
   DataFrame.std
   DataFrame.sum
   DataFrame.value_counts
   DataFrame.var

Reindexing / selection / label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.add_prefix
   DataFrame.add_suffix
   DataFrame.drop_duplicates
   DataFrame.duplicated
   DataFrame.head
   DataFrame.rename
   DataFrame.rename_axis
   DataFrame.reset_index
   DataFrame.sample
   DataFrame.set_axis
   DataFrame.tail

.. _generated.dataframe.missing:

Missing data handling
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.isna
   DataFrame.isnull
   DataFrame.notna
   DataFrame.notnull

Reshaping, sorting, transposing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.sort_values
   DataFrame.sort_index

Combining / joining / merging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.join
   DataFrame.merge

.. _generated.dataframe.plotting:

Plotting
~~~~~~~~
``DataFrame.plot`` is both a callable method and a namespace attribute for
specific plotting methods of the form ``DataFrame.plot.<kind>``.

.. autosummary::
   :toctree: generated/
   :template: accessor_callable.rst

   DataFrame.plot

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   DataFrame.plot.area
   DataFrame.plot.bar
   DataFrame.plot.barh
   DataFrame.plot.box
   DataFrame.plot.density
   DataFrame.plot.hexbin
   DataFrame.plot.hist
   DataFrame.plot.kde
   DataFrame.plot.line
   DataFrame.plot.pie
   DataFrame.plot.scatter

.. _generated.dataframe.io:

Serialization / IO / conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.to_odps_table
