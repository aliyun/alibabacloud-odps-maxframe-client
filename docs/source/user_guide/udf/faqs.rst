.. _udf_faqs:

FAQs
====

Start with the error message and the point at which it occurs. Errors while
building an expression can happen on the client; errors after submission are
reported in Logview. The cases below explain what to check and how to proceed.

Client hangs or triggers external calls while building the expression
---------------------------------------------------------------------

If calling ``apply_chunk``, ``apply``, or ``map`` loads a model, sends a request,
or raises ``ValueError`` / ``KeyError`` before ``execute()``, check whether
MaxFrame is inferring the result type. Inference calls the function on the
client with synthetic values that may not resemble real input.

Declare the result using the arguments supported by that interface. For
DataFrame and Series ``apply_chunk``, a complete return annotation or explicit
type arguments with ``skip_infer=True`` avoids the trial call. See
:ref:`udf_result_types` for the available forms. If the problem remains with
inference disabled, inspect code that runs outside the UDF, such as imports
or model construction.

Why do the returned columns, types, or index not match?
-------------------------------------------------------

For errors about column names or dtypes, run the function on a small pandas
sample and compare the result with its declaration. Check the column order,
each column's dtype, and the Series name where applicable. An empty branch
must return the same columns and dtypes, rather than an untyped
``pd.DataFrame()``.

If a later join or assignment produces unexpected ``NaN`` values, also check
the index. A new DataFrame built from arrays starts its index at zero unless
one is provided. For one result row per input row, construct it with
``index=batch.index``. For filtered or expanded results, follow
:ref:`udf_result_index`.

Why does the remote job report ModuleNotFoundError?
---------------------------------------------------

Check which package is missing in Logview StdErr. Packages installed on the
client are not automatically installed in the remote environment. Add the
package and version with ``@with_python_requirements(...)``, or use a
registered image that contains it. See :ref:`udf_config_packages`.

.. _udf_oom:

How can an OOM be resolved?
---------------------------

First check the failed task's logs in :ref:`udf_debugging`. An OOM-related
error can contain the following lines:

.. code-block:: text

   ExitCode: -1
   ExitError: signal: killed
   The process may be OOM killed.

``signal: killed`` alone does not prove an OOM. Compare the error with the
task's memory usage and retry history. For MCSQL workers, see
:ref:`logview_fuxi_sensor`; for DPE, inspect the matching SubStep and its Sensor
entry when available. Compare failed tasks with normal tasks and the largest
input groups to distinguish skew from generally insufficient memory.

Choose the adjustment that matches the memory use:

#. **Reduce the input held by a task.** In DPE, lower the applicable read-table
   chunk setting or use ``rechunk`` / ``rebalance`` before the UDF. DataFrame
   ``rechunk`` is effective only in DPE. In MCSQL, ``chunk_bytes`` sets a SQL
   split hint and ``chunk_size`` is ignored; SPE ignores both read-table
   settings. See :ref:`udf_source_chunk_settings` for the table-specific rules.
   For grouped UDFs, smaller source chunks do not limit group size. Inspect
   the largest groups; see :ref:`udf_batch_rows` for grouped batching.
#. **Reduce per-call temporary memory**: Set or lower ``batch_rows`` in
   ``apply_chunk`` or ``map_reduce`` so each Python call allocates smaller
   intermediate pandas/NumPy objects. Note that ``batch_rows`` alone does not
   shrink the task's input chunk. See :ref:`udf_batch_rows`.
#. **Check group skew and task memory.** Count rows by the grouping keys, then
   sort the counts to find the largest groups and compare them with typical
   groups. A few very large groups can cause OOM failures even when source
   chunks are small. If a model or a complete group still needs more memory,
   request it with ``@with_running_options(memory="16GiB")`` (see
   :ref:`udf_tuning_resources` and :ref:`udf_configuration_guide`).

If most DPE CPU UDF tasks succeed and only a few run out of memory, enable
:ref:`elastic memory <user_guide_elastic_memory>` with ``memory_limit``. Those
tasks can retry with more memory, up to the configured limit.

.. _udf_elastic_memory_not_expanding:

Why does memory not increase after setting memory_limit?
--------------------------------------------------------

Check the failed task in Logview, then work through the following:

#. **Check the engine and resources.** Elastic memory applies to DPE CPU UDFs.
   MCSQL, SPE, and GU tasks do not use it. The SDK and DPE runtime must support
   the feature.
#. **Check the effective settings.** Combine the decorator with the shared
   defaults. ``memory_limit`` must be greater than ``memory`` to enable
   expansion; equal values disable it. The cluster may cap the effective
   limit. See :ref:`udf_config_elastic_overrides` for inheritance examples.
#. **Check the failure type and retry history.** Only a recognized OOM
   increases memory. A network or other ordinary error keeps the current
   allocation. If the task has reached the memory limit and still runs out
   of memory, it fails. See :ref:`udf_elastic_memory_observation` for the
   Logview labels and :ref:`udf_oom` for reducing memory use.
#. **Check task restrictions.** The task must allow retries. Elastic memory
   is unavailable with ``step.skip_task_resources=True``, concurrency limits,
   or gang scheduling. Platform protection rules can also stop retries before
   the limit is reached. See :ref:`udf_elastic_memory_limits`.
