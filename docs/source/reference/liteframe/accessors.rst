.. _generated.liteframe_accessors:

Accessors
=========
.. currentmodule:: maxframe.liteframe

LiteFrame provides dtype-specific methods under various accessors.
These are separate namespaces within :class:`LiteFrame` that only apply
to specific data types.

=========================== =================================
Data Type                   Accessor
=========================== =================================
Datetime, Timedelta         :ref:`dt <generated.liteframe.dt>`
String                      :ref:`str <generated.liteframe.str>`
List                        :ref:`list <generated.liteframe.list>`
Dict                        :ref:`dict <generated.liteframe.dict>`
Struct                      :ref:`struct <generated.liteframe.struct>`
=========================== =================================

.. _generated.liteframe.str:

String handling
~~~~~~~~~~~~~~~

``LiteFrame.str`` can be used to access the values of the columns as
strings and apply several methods to it. These can be accessed like
``LiteFrame.str.<function/property>``.

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   LiteFrame.str.capitalize
   LiteFrame.str.center
   LiteFrame.str.contains
   LiteFrame.str.count
   LiteFrame.str.endswith
   LiteFrame.str.extract
   LiteFrame.str.find
   LiteFrame.str.fullmatch
   LiteFrame.str.isalnum
   LiteFrame.str.isalpha
   LiteFrame.str.isdecimal
   LiteFrame.str.isdigit
   LiteFrame.str.islower
   LiteFrame.str.isnumeric
   LiteFrame.str.isspace
   LiteFrame.str.istitle
   LiteFrame.str.isupper
   LiteFrame.str.len
   LiteFrame.str.ljust
   LiteFrame.str.lower
   LiteFrame.str.lstrip
   LiteFrame.str.match
   LiteFrame.str.pad
   LiteFrame.str.partition
   LiteFrame.str.removeprefix
   LiteFrame.str.removesuffix
   LiteFrame.str.repeat
   LiteFrame.str.replace
   LiteFrame.str.rjust
   LiteFrame.str.rpartition
   LiteFrame.str.rstrip
   LiteFrame.str.rsplit
   LiteFrame.str.slice
   LiteFrame.str.split
   LiteFrame.str.startswith
   LiteFrame.str.strip
   LiteFrame.str.title
   LiteFrame.str.upper
   LiteFrame.str.zfill
   LiteFrame.str.normalize

.. _generated.liteframe.dt:

Datetimelike handling
~~~~~~~~~~~~~~~~~~~~~

``LiteFrame.dt`` can be used to access the values of the columns as
datetimelike and return several properties.
These can be accessed like ``LiteFrame.dt.<property>``.

Datetime properties
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_attribute.rst

   LiteFrame.dt.date
   LiteFrame.dt.time
   LiteFrame.dt.year
   LiteFrame.dt.month
   LiteFrame.dt.day
   LiteFrame.dt.hour
   LiteFrame.dt.minute
   LiteFrame.dt.second
   LiteFrame.dt.microsecond
   LiteFrame.dt.nanosecond
   LiteFrame.dt.dayofweek
   LiteFrame.dt.dayofyear
   LiteFrame.dt.weekofyear
   LiteFrame.dt.quarter
   LiteFrame.dt.days_in_month
   LiteFrame.dt.daysinmonth
   LiteFrame.dt.is_month_start
   LiteFrame.dt.is_month_end
   LiteFrame.dt.is_quarter_start
   LiteFrame.dt.is_quarter_end
   LiteFrame.dt.is_year_start
   LiteFrame.dt.is_year_end
   LiteFrame.dt.is_leap_year

Timedelta properties
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_attribute.rst

   LiteFrame.dt.days
   LiteFrame.dt.seconds
   LiteFrame.dt.microseconds
   LiteFrame.dt.nanoseconds
   LiteFrame.dt.components

Datetime methods
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   LiteFrame.dt.strftime
   LiteFrame.dt.day_name
   LiteFrame.dt.month_name
   LiteFrame.dt.normalize
   LiteFrame.dt.round
   LiteFrame.dt.floor
   LiteFrame.dt.ceil
   LiteFrame.dt.total_seconds
   LiteFrame.dt.isocalendar

.. _generated.liteframe.list:

List handling
~~~~~~~~~~~~~

``LiteFrame.list`` can be used to access the methods of the columns with
list values. These can be accessed like ``LiteFrame.list.<method>``.

List methods
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   LiteFrame.list.__getitem__
   LiteFrame.list.get
   LiteFrame.list.len
   LiteFrame.list.flatten

.. _generated.liteframe.dict:

Dict handling
~~~~~~~~~~~~~

``LiteFrame.dict`` can be used to access the methods of the columns with
dict values. These can be accessed like ``LiteFrame.dict.<method>``.

Dict methods
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   LiteFrame.dict.__getitem__
   LiteFrame.dict.contains
   LiteFrame.dict.get
   LiteFrame.dict.len
   LiteFrame.dict.remove

.. _generated.liteframe.struct:

Struct handling
~~~~~~~~~~~~~~~


Struct properties
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_attribute.rst

   LiteFrame.struct.dtypes

Struct methods
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   LiteFrame.struct.field
   LiteFrame.struct.explode
