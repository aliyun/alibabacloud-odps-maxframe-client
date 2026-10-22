.. _generated.liteframe:

Frame methods
=============
.. currentmodule:: maxframe.liteframe

LiteFrame is a performance-dedicated tabular data object with no index semantics.
Column names are flat strings, single-column selection returns a LiteFrame (not a
Series), and all operations are column-based with no implicit label alignment.
Arrow dtypes are used throughout so that nullable integer columns are preserved
rather than promoted to float. Results are lazy — call ``.execute()`` to materialize.

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   LiteFrame

   LiteFrame.from_local_df
   LiteFrame.from_tensor

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Axes**

.. autosummary::
   :toctree: generated/

   LiteFrame.columns

.. autosummary::
   :toctree: generated/

   LiteFrame.dtypes
   LiteFrame.ndim
   LiteFrame.shape
   LiteFrame.values

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   LiteFrame.astype
   LiteFrame.to_pandas
   LiteFrame.to_tensor

Indexing, iteration
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   LiteFrame.eval
   LiteFrame.query
   LiteFrame.head
   LiteFrame.iloc

Binary operator functions
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   LiteFrame.add
   LiteFrame.sub
   LiteFrame.mul
   LiteFrame.div
   LiteFrame.floordiv
   LiteFrame.mod
   LiteFrame.pow
   LiteFrame.radd
   LiteFrame.rsub
   LiteFrame.rmul
   LiteFrame.rdiv
   LiteFrame.rfloordiv
   LiteFrame.rmod
   LiteFrame.rpow
   LiteFrame.lshift
   LiteFrame.rshift
   LiteFrame.rlshift
   LiteFrame.rrshift
   LiteFrame.lt
   LiteFrame.gt
   LiteFrame.le
   LiteFrame.ge
   LiteFrame.ne
   LiteFrame.eq

Function application, GroupBy & window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   LiteFrame.apply
   LiteFrame.apply_chunk
   LiteFrame.expanding
   LiteFrame.flat_apply
   LiteFrame.groupby
   LiteFrame.map
   LiteFrame.rolling
   LiteFrame.transform
   LiteFrame.transform_chunk

Computations / descriptive stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   LiteFrame.agg
   LiteFrame.all
   LiteFrame.between
   LiteFrame.case_when
   LiteFrame.mask
   LiteFrame.where
   LiteFrame.any
   LiteFrame.clip
   LiteFrame.count
   LiteFrame.cummax
   LiteFrame.cummin
   LiteFrame.cumprod
   LiteFrame.cumsum
   LiteFrame.diff
   LiteFrame.kurt
   LiteFrame.max
   LiteFrame.mean
   LiteFrame.median
   LiteFrame.min
   LiteFrame.nunique
   LiteFrame.pct_change
   LiteFrame.prod
   LiteFrame.round
   LiteFrame.sem
   LiteFrame.size
   LiteFrame.skew
   LiteFrame.std
   LiteFrame.sum
   LiteFrame.var

Reindexing / selection / label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   LiteFrame.add_prefix
   LiteFrame.add_suffix
   LiteFrame.assign
   LiteFrame.drop
   LiteFrame.drop_duplicates
   LiteFrame.duplicated
   LiteFrame.rename
   LiteFrame.select

Missing data handling
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   LiteFrame.dropna
   LiteFrame.fillna
   LiteFrame.isna
   LiteFrame.isnull
   LiteFrame.notna
   LiteFrame.notnull
   LiteFrame.replace

Reshaping, sorting
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   LiteFrame.explode
   LiteFrame.rebalance
   LiteFrame.rechunk
   LiteFrame.sort

Combining / joining / merging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   LiteFrame.merge

Time series-related
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   LiteFrame.bfill
   LiteFrame.ffill
   LiteFrame.shift
