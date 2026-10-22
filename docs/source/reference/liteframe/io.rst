.. _generated.liteframe_io:

Input/output
============
.. currentmodule:: maxframe.liteframe

LiteFrame IO functions read and write tabular data without index-specific
metadata. All read functions use Arrow dtypes (``dtype_backend="pyarrow"``)
by default. Write functions have no ``index`` or ``index_label`` parameters
because LiteFrame has no index semantics.

Flat file
~~~~~~~~~
.. autosummary::
   :toctree: generated/

   read_csv
   LiteFrame.to_csv

JSON
~~~~
.. autosummary::
   :toctree: generated/

   read_json
   LiteFrame.to_json

MaxCompute
~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   read_odps_table
   LiteFrame.to_odps_table

Parquet
~~~~~~~
.. autosummary::
   :toctree: generated/

   read_parquet
   LiteFrame.to_parquet

Lance
~~~~~
.. autosummary::
   :toctree: generated/

   read_lance
   LiteFrame.to_lance
