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
   DataFrame.memory_usage
   DataFrame.ndim
   DataFrame.select_dtypes
   DataFrame.shape

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.astype

Indexing, iteration
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.at
   DataFrame.head
   DataFrame.iat
   DataFrame.iloc
   DataFrame.insert
   DataFrame.loc
   DataFrame.mask
   DataFrame.pop
   DataFrame.query
   DataFrame.tail
   DataFrame.xs
   DataFrame.where

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
   DataFrame.combine_first

Function application, GroupBy & window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.apply
   DataFrame.applymap
   DataFrame.agg
   DataFrame.aggregate
   DataFrame.ewm
   DataFrame.expanding
   DataFrame.groupby
   DataFrame.map
   DataFrame.rolling
   DataFrame.transform

.. _generated.dataframe.stats:

Computations / descriptive stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.abs
   DataFrame.all
   DataFrame.any
   DataFrame.clip
   DataFrame.count
   DataFrame.corr
   DataFrame.corrwith
   DataFrame.cov
   DataFrame.describe
   DataFrame.diff
   DataFrame.eval
   DataFrame.max
   DataFrame.mean
   DataFrame.median
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
   DataFrame.align
   DataFrame.drop
   DataFrame.drop_duplicates
   DataFrame.droplevel
   DataFrame.duplicated
   DataFrame.filter
   DataFrame.head
   DataFrame.idxmax
   DataFrame.idxmin
   DataFrame.reindex
   DataFrame.reindex_like
   DataFrame.rename
   DataFrame.rename_axis
   DataFrame.reset_index
   DataFrame.sample
   DataFrame.set_axis
   DataFrame.set_index
   DataFrame.take
   DataFrame.truncate

.. _generated.dataframe.missing:

Missing data handling
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.dropna
   DataFrame.fillna
   DataFrame.isna
   DataFrame.isnull
   DataFrame.notna
   DataFrame.notnull

Reshaping, sorting, transposing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.melt
   DataFrame.nlargest
   DataFrame.nsmallest
   DataFrame.pivot
   DataFrame.pivot_table
   DataFrame.reorder_levels
   DataFrame.sort_values
   DataFrame.sort_index
   DataFrame.swaplevel
   DataFrame.stack
   DataFrame.unstack

Combining / comparing / joining / merging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.append
   DataFrame.assign
   DataFrame.compare
   DataFrame.join
   DataFrame.merge
   DataFrame.update

Time series-related
-------------------
.. autosummary::
   :toctree: generated/

   DataFrame.first_valid_index
   DataFrame.last_valid_index
   DataFrame.shift
   DataFrame.tshift

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

   DataFrame.from_dict
   DataFrame.from_records
   DataFrame.to_csv
   DataFrame.to_odps_table
   DataFrame.to_pandas

.. _generated.dataframe.mf:

MaxFrame Extensions
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   DataFrame.mf.apply_chunk
   DataFrame.mf.collect_kv
   DataFrame.mf.extract_kv
   DataFrame.mf.flatmap
   DataFrame.mf.map_reduce
   DataFrame.mf.rebalance
   DataFrame.mf.reshuffle

``DataFrame.mf`` provides methods unique to MaxFrame. These methods are collated from application
scenarios in MaxCompute and these can be accessed like ``DataFrame.mf.<function/property>``.
