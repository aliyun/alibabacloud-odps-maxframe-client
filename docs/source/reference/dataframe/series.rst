Series
======
.. currentmodule:: maxframe.dataframe

Constructor
-----------
.. autosummary::
   :toctree: generated/

   Series

Attributes
----------
**Axes**

.. autosummary::
   :toctree: generated/

   Series.index

.. autosummary::
   :toctree: generated/

   Series.dtype
   Series.hasnans
   Series.memory_usage
   Series.ndim
   Series.name
   Series.shape
   Series.T

Conversion
----------
.. autosummary::
   :toctree: generated/

   Series.astype
   Series.copy
   Series.to_frame

Index, iteration
----------------
.. autosummary::
   :toctree: generated/

   Series.at
   Series.iat
   Series.iloc
   Series.loc
   Series.mask
   Series.xs
   Series.where

Binary operator functions
-------------------------
.. autosummary::
   :toctree: generated/

   Series.add
   Series.sub
   Series.mul
   Series.div
   Series.truediv
   Series.floordiv
   Series.mod
   Series.pow
   Series.radd
   Series.rsub
   Series.rmul
   Series.rdiv
   Series.rtruediv
   Series.rfloordiv
   Series.rmod
   Series.rpow
   Series.lt
   Series.gt
   Series.le
   Series.ge
   Series.ne
   Series.eq
   Series.combine_first

Function application, groupby & window
--------------------------------------
.. autosummary::
   :toctree: generated/

   Series.apply
   Series.agg
   Series.aggregate
   Series.ewm
   Series.expanding
   Series.groupby
   Series.map
   Series.rolling
   Series.transform

.. _generated.series.stats:

Computations / descriptive stats
--------------------------------
.. autosummary::
   :toctree: generated/

   Series.abs
   Series.all
   Series.any
   Series.between
   Series.clip
   Series.corr
   Series.count
   Series.cov
   Series.describe
   Series.is_monotonic_increasing
   Series.is_monotonic_decreasing
   Series.is_unique
   Series.max
   Series.mean
   Series.min
   Series.median
   Series.nlargest
   Series.nsmallest
   Series.nunique
   Series.prod
   Series.product
   Series.quantile
   Series.round
   Series.sem
   Series.std
   Series.sum
   Series.unique
   Series.value_counts
   Series.var

Reindexing / selection / label manipulation
-------------------------------------------
.. autosummary::
   :toctree: generated/

   Series.add_prefix
   Series.add_suffix
   Series.align
   Series.case_when
   Series.drop
   Series.drop_duplicates
   Series.droplevel
   Series.filter
   Series.head
   Series.idxmax
   Series.idxmin
   Series.isin
   Series.reindex
   Series.reindex_like
   Series.rename
   Series.reset_index
   Series.sample
   Series.set_axis
   Series.take
   Series.truncate

Missing data handling
---------------------
.. autosummary::
   :toctree: generated/

   Series.dropna
   Series.fillna
   Series.isna
   Series.notna
   Series.dropna
   Series.fillna

Reshaping, sorting
------------------
.. autosummary::
   :toctree: generated/

   Series.argmax
   Series.argmin
   Series.argsort
   Series.explode
   Series.reorder_levels
   Series.sort_values
   Series.sort_index
   Series.swaplevel
   Series.unstack

Combining / comparing / joining / merging
-----------------------------------------
.. autosummary::
   :toctree: generated/

   Series.append
   Series.compare
   Series.update

Time Series-related
-------------------
.. autosummary::
   :toctree: generated/

   Series.first_valid_index
   Series.last_valid_index
   Series.shift
   Series.tshift

Accessors
---------

Pandas provides dtype-specific methods under various accessors.
These are separate namespaces within :class:`Series` that only apply
to specific data types.

=========================== =================================
Data Type                   Accessor
=========================== =================================
Datetime, Timedelta, Period :ref:`dt <generated.series.dt>`
String                      :ref:`str <generated.series.str>`
Dict                        :ref:`dict <generated.series.dict>`
=========================== =================================

.. _generated.series.dt:

Datetimelike properties
~~~~~~~~~~~~~~~~~~~~~~~

``Series.dt`` can be used to access the values of the series as
datetimelike and return several properties.
These can be accessed like ``Series.dt.<property>``.

Datetime properties
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_attribute.rst

   Series.dt.date
   Series.dt.time
   Series.dt.timetz
   Series.dt.year
   Series.dt.month
   Series.dt.day
   Series.dt.hour
   Series.dt.minute
   Series.dt.second
   Series.dt.microsecond
   Series.dt.nanosecond
   Series.dt.week
   Series.dt.weekofyear
   Series.dt.dayofweek
   Series.dt.weekday
   Series.dt.dayofyear
   Series.dt.quarter
   Series.dt.is_month_start
   Series.dt.is_month_end
   Series.dt.is_quarter_start
   Series.dt.is_quarter_end
   Series.dt.is_year_start
   Series.dt.is_year_end
   Series.dt.is_leap_year
   Series.dt.daysinmonth
   Series.dt.days_in_month

Datetime methods
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   Series.dt.to_period
   Series.dt.to_pydatetime
   Series.dt.tz_localize
   Series.dt.tz_convert
   Series.dt.normalize
   Series.dt.strftime
   Series.dt.round
   Series.dt.floor
   Series.dt.ceil
   Series.dt.month_name
   Series.dt.day_name


.. _generated.series.str:

String handling
~~~~~~~~~~~~~~~

``Series.str`` can be used to access the values of the series as
strings and apply several methods to it. These can be accessed like
``Series.str.<function/property>``.

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   Series.str.capitalize
   Series.str.contains
   Series.str.count
   Series.str.endswith
   Series.str.find
   Series.str.len
   Series.str.ljust
   Series.str.lower
   Series.str.lstrip
   Series.str.pad
   Series.str.repeat
   Series.str.replace
   Series.str.rfind
   Series.str.rjust
   Series.str.rstrip
   Series.str.slice
   Series.str.startswith
   Series.str.strip
   Series.str.swapcase
   Series.str.title
   Series.str.translate
   Series.str.upper
   Series.str.zfill
   Series.str.isalnum
   Series.str.isalpha
   Series.str.isdigit
   Series.str.isspace
   Series.str.islower
   Series.str.isupper
   Series.str.istitle
   Series.str.isnumeric
   Series.str.isdecimal

..
    The following is needed to ensure the generated pages are created with the
    correct template (otherwise they would be created in the Series/Index class page)

..
    .. autosummary::
       :toctree: generated/
       :template: accessor.rst

       Series.str
       Series.dt

.. _generated.series.dict:

Dict properties
~~~~~~~~~~~~~~~~~~~~~~~

``Series.dict`` can be used to access the methods of the series with dict values.
These can be accessed like ``Series.dict.<method>``.


Dict methods
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   Series.dict.__getitem__
   Series.dict.__setitem__
   Series.dict.contains
   Series.dict.get
   Series.dict.len
   Series.dict.remove

.. _generated.series.list:

List properties
~~~~~~~~~~~~~~~~~~~~~~~

``Series.list`` can be used to access the methods of the series with list values.
These can be accessed like ``Series.list.<method>``.


List methods
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   Series.list.__getitem__
   Series.list.len

Struct properties
~~~~~~~~~~~~~~~~~~~~~~~

``Series.struct`` can be used to access the methods of the series with struct values.
These can be accessed like ``Series.struct.<method>``.


Struct methods
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   Series.struct.dtypes
   Series.struct.field


Plotting
--------
``Series.plot`` is both a callable method and a namespace attribute for
specific plotting methods of the form ``Series.plot.<kind>``.

.. autosummary::
   :toctree: generated/
   :template: accessor_callable.rst

   Series.plot

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   Series.plot.area
   Series.plot.bar
   Series.plot.barh
   Series.plot.box
   Series.plot.density
   Series.plot.hist
   Series.plot.kde
   Series.plot.line
   Series.plot.pie

.. _generated.series.mf:

MaxFrame Extensions
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   Series.mf.apply_chunk
   Series.mf.flatmap
   Series.mf.flatjson

``Series.mf`` The Series.mf provides methods unique to MaxFrame. These methods are collated from application
scenarios in MaxCompute and these can be accessed like ``Series.mf.<function/property>``.
