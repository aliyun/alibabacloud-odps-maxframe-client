.. _udf_complete_example:


A complete apply_chunk job
==========================

This example reads application logs from ``source_table``, with ``id`` as the
DataFrame index. The UDF extracts fields and flags slow requests in batches.
The job writes the result to ``target_table`` and reads back a few rows.

Before running the example, set the MaxCompute credentials, project, and
endpoint in the environment variables used below. Replace the source and
target table names with the actual names.

The source table must contain ``id`` and ``raw_log`` columns. ``id`` identifies
each record, and ``raw_log`` contains the log text. For example, the
illustrative input ``id=101`` and
``raw_log="[INFO] latency=800ms"`` produces ``level="INFO"``,
``latency_ms=800.0``, and ``is_slow=True``, with ``101`` kept as the row index.

The example uses ``overwrite=True``, which replaces existing data in
``target_table``, and ``index=True`` so the ``id`` index is written as a column.

.. code-block:: python

   import os

   import pandas as pd
   from odps import ODPS

   import maxframe.dataframe as md
   from maxframe.session import new_session
   from maxframe.udf import with_running_options


   o = ODPS(
       os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
       os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
       project=os.environ["ODPS_PROJECT"],
       endpoint=os.environ["ODPS_ENDPOINT"],
   )
   o.to_global()  # to_odps_table uses the PyODPS default connection.
   session = new_session(o)
   print(f"Logview URL: {session.get_logview_address()}")


   @with_running_options(cpu=2, memory="8GiB")
   def parse_log_batch(chunk: pd.DataFrame) -> pd.DataFrame[
       {"level": "object", "latency_ms": "float64", "is_slow": "bool"}
   ]:
       extracted = chunk["raw_log"].str.extract(
           r"^\[(?P<level>[A-Z]+)\]\s+latency=(?P<latency_ms>\d+(?:\.\d+)?)ms"
       )
       latency = pd.to_numeric(extracted["latency_ms"], errors="coerce").astype("float64")
       return pd.DataFrame(
           {
               "level": extracted["level"].fillna("UNKNOWN"),
               "latency_ms": latency,
               "is_slow": latency >= 500.0,
           },
           index=chunk.index,
       )


   try:
       df = md.read_odps_table("source_table", odps_entry=o, index_col="id")
       result = df.mf.apply_chunk(parse_log_batch, batch_rows=2000, index=df.index)
       md.to_odps_table(result, "target_table", overwrite=True, index=True).execute(
           session=session
       )
       saved = md.read_odps_table("target_table", odps_entry=o, index_col="id")
       print(saved.head(5).execute(session=session).fetch())
   finally:
       session.destroy()

If the source contains only the illustrative record above, the read-back
result is:

.. code-block:: text

       level  latency_ms  is_slow
   id
   101  INFO       800.0     True

When adapting this example, check these settings:

* ``odps_entry=o`` supplies the connection when reading each table.
  ``o.to_global()`` sets the PyODPS default used by ``to_odps_table``.
  Creating a MaxFrame session does not set that default.

* ``new_session(o)`` creates the MaxFrame session, and
  ``get_logview_address()`` returns the Logview link for checking progress and
  errors.

* ``@with_running_options(cpu=2, memory="8GiB")`` requests 2 CPUs and 8 GiB of
  memory for each task running the function while following the job's default
  execution engine. Adjust these values for the workload; see
  :ref:`udf_tuning_resources`.

* The return annotation declares the output columns and dtypes. The function
  preserves the input ``id`` index; see :ref:`udf_result_index`.

* Reading the table and calling ``apply_chunk`` build the calculation.
  Executing ``to_odps_table`` submits the calculation and table write.
  The final read checks up to five saved rows; ``fetch()`` returns them as a
  pandas DataFrame. Row order is not guaranteed.
