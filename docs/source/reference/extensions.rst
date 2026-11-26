.. _reference_extensions:

Extension APIs
==============

Accessor registration
---------------------

Custom accessor namespaces are registered through the ``api`` package of the
data type they belong to. Those modules are shortcuts onto the registrators
defined in the data-type core modules, next to the type tuples they install on;
the registry itself and ``register_accessor`` (the generic entry point taking an
arbitrary class tuple) live in ``maxframe.core.accessor``.

An accessor is an ordinary class that is instantiated with the object the
attribute was read from, following the pandas convention: it raises
``AttributeError`` from ``__init__`` when the object is not supported.

.. currentmodule:: maxframe.dataframe.api.extensions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   register_dataframe_accessor
   register_series_accessor
   register_index_accessor

.. currentmodule:: maxframe.liteframe.api.extensions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   register_liteframe_accessor

.. currentmodule:: maxframe.tensor.api.extensions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   register_tensor_accessor
