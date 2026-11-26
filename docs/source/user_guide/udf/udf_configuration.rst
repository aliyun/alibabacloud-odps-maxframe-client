.. _udf_configuration:

UDF configuration
=================

Use decorators from ``maxframe.udf`` when a UDF needs more memory, third-party
Python packages, or access to OSS files. When multiple UDFs need the same
resources, image, or network whitelists, set defaults with
``maxframe.config.options``. Complete decorator signatures and parameter
definitions are in the :doc:`UDF API reference <../../reference/udf>`.

.. _udf_config_engine_support:

Engine support
--------------

The following table covers the settings used in this guide. The selected
engine must also support the UDF operation.

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Configuration
     - DPE
     - MCSQL
     - SPE
   * - ``with_running_options``: ``cpu``, ``memory``; corresponding shared defaults
     - Task resources
     - UDF worker resources
     - Ignored
   * - ``with_running_options``: ``engine``
     - ``"DPE"``
     - ``"MCSQL"``
     - ``"SPE"``
   * - ``memory_limit`` in the decorator or shared defaults
     - Eligible CPU UDFs
     - Not supported
     - Not supported
   * - ``gu``, ``gu_type``, ``gu_quota``; shared ``gpu`` and ``session.gu_quota_name``
     - GPU tasks
     - Not supported
     - Not supported
   * - ``with_python_requirements`` (all build options); ``with_resources``
     - Supported
     - Supported
     - Not loaded
   * - ``with_fs_mount``; ``with_network_options``; ``with_image_options``
     - Supported
     - Not supported
     - Not supported
   * - ``python_tag``
     - Supported
     - Supported
     - Remote runtime only
   * - :ref:`Memory per CPU limit <udf_config_resources>`
     - Adjusts CPU requests
     - Adjusts CPU requests
     - Client validation only
   * - ``session.image_name``; public/internal whitelists in ``dpe.settings``
     - Supported
     - Not supported
     - Not supported

Shared names in this table omit the ``options.`` prefix.
``with_resource_libraries`` is an alias of ``with_resources``.
Mounts, network settings, and custom images do not select DPE automatically;
the examples below select it explicitly. GPU tasks need an available quota.
For elastic-memory requirements, see :ref:`user_guide_elastic_memory`.

SPE runs UDFs in one Python process. The CPU and memory settings above do not
resize that process. Packages and files must already be available in its
environment; the UDF decorators do not install or load them for SPE.
``python_tag`` selects the remote SPE runtime; local execution uses the current
Python process.
Unsupported settings do not provide the corresponding feature on that engine.
Decorator and resource validation still applies before execution, so an invalid
combination can raise an error even when the engine does not use the settings.

Configure one user-defined function
-----------------------------------

Before passing a function that reads files or calls a service to MaxFrame,
declare its result type to avoid a trial call on the client.
See :ref:`udf_result_types`.

.. _udf_config_resources:

.. _with-running-options:

Execution engine and resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``with_running_options`` to select an engine or adjust the resources for
a UDF. Both DPE and MCSQL support ``cpu`` and ``memory``. ``memory_limit``,
``gu``, ``gu_type``, and ``gu_quota`` apply only to DPE, as listed in
:ref:`udf_config_engine_support`.

DPE uses ``cpu`` and ``memory`` as resource requests for each function task.
MCSQL applies them to UDF worker settings and also calculates compute units
(CU), using one CPU and 4 GiB of memory per CU. For example, ``cpu=2`` and
``memory="12GiB"`` produce a CU setting of 3.

Integer memory values use GiB; strings can specify units, such as ``"512MiB"``.
Use ``gu`` and ``gu_quota`` to request GPU units and select a DPE GPU quota.
For a Spot GU quota, see ``gu_type`` and ``gu_quota`` in
:func:`maxframe.udf.with_running_options`.

Use one ``with_running_options`` decorator per UDF. An outer decorator that
sets an engine or resource option replaces the inner engine and whole resource
dictionary. Fields are not merged between decorators; omitted fields inherit
the shared defaults. An empty decorator, or one setting only ``image_name``,
keeps the existing engine and resources. ``gu`` or ``gu_type`` cannot be
combined with ``cpu`` or ``memory``.

If the requested memory per CPU exceeds
``options.function.allowed_max_memory_cpu_ratio``, MaxFrame increases the CPU
request before submission and emits a warning.

The combined example below uses ``cpu=2`` and ``memory="8GiB"``. For memory
sizing and CPU adjustments, see :ref:`udf_tuning_resources`.

.. _udf_config_elastic_memory:

Elastic memory
^^^^^^^^^^^^^^

``memory_limit`` sets the maximum allocation for eligible DPE CPU UDF
retries after OOM. Set it above ``memory`` to enable expansion. See
:ref:`user_guide_elastic_memory` for examples, shared defaults, override rules,
and troubleshooting.

.. _udf_config_packages:

.. _with-python-requirements:

Python packages
~~~~~~~~~~~~~~~

If a UDF imports a package that is missing from the remote environment, add
it with ``with_python_requirements``. Installing the package on the client
does not install it in the remote environment.

Before the function task runs, MaxFrame submits a PythonPack task, which builds
or reuses the remote Python package environment. A matching build has the same
requirement set and build options. ``prefer_binary=True`` tells PythonPack to
prefer binary distributions.

Available wheels also depend on the remote Python ABI. See
:ref:`udf_config_python` for ``options.python_tag`` when diagnosing a package
build or wheel compatibility failure.

.. _udf_config_files:

.. _with-resources:

MaxCompute resource files
~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``with_resources`` for files such as label lists or configuration files
uploaded as MaxCompute resources.

The combined example below opens ``labels.txt`` from the function's working
directory. See :func:`maxframe.udf.with_resources` for resource name formats.

.. _udf_config_mounts:

.. _with-fs-mount:

OSS files
~~~~~~~~~

If the function or a library reads files by local path, use ``with_fs_mount``
to mount an OSS path in DPE. The UDF can then read the files through the mount
path. MCSQL does not support these mounts; see :ref:`udf_config_engine_support`.

.. code-block:: python

   import json
   from pathlib import Path

   from maxframe.udf import with_fs_mount, with_running_options


   OSS_STORAGE_OPTIONS = {
       "role_arn": "acs:ram::<account-id>:role/<role-name>",
   }

   @with_running_options(engine="DPE")
   @with_fs_mount(
       path="oss://oss-cn-hangzhou.aliyuncs.com/bucket/images",
       mount_path="/mnt/images",
       storage_options=OSS_STORAGE_OPTIONS,
   )
   @with_fs_mount(
       path="oss://oss-cn-hangzhou.aliyuncs.com/bucket/models",
       mount_path="/mnt/models",
       storage_options=OSS_STORAGE_OPTIONS,
   )
   def process_mounted_files(chunk):
       config = json.loads(
           Path("/mnt/models/config.json").read_text(encoding="utf-8")
       )
       scale = float(config.get("scale", 1.0))
       sizes = [
           Path("/mnt/images", name).stat().st_size * scale
           for name in chunk["file_name"]
       ]
       return chunk[["id"]].assign(weighted_size=sizes)

When mounting multiple OSS paths on one UDF, use the same ``storage_options``
for every ``with_fs_mount`` decorator.

For AccessKey authentication and the full ``storage_options`` contract, see
:func:`maxframe.udf.with_fs_mount`.

.. _udf_config_network:

.. _with-network-options:

Network access
~~~~~~~~~~~~~~

When a UDF calls a remote service, configure network access with
``with_network_options``. Choose the setting for the public network,
Alibaba Cloud internal network, or VPC where the service is located. These
settings apply to DPE only; see :ref:`udf_config_engine_support`.

.. code-block:: python

   import json
   import urllib.request

   from maxframe.udf import with_network_options, with_running_options


   @with_running_options(engine="DPE")
   @with_network_options(
       public_whitelist=["<approved-public-endpoint>"],
       internal_whitelist=["<approved-internal-endpoint>"],
       vpc_network_link="<configured-network-link-name>",
   )
   def access_remote_services(chunk):
       payload = json.dumps({"texts": chunk["text"].tolist()}).encode("utf-8")
       req = urllib.request.Request(
           "http://<approved-internal-endpoint>/v1/score",
           data=payload,
           headers={"Content-Type": "application/json"},
       )
       with urllib.request.urlopen(req, timeout=5) as resp:
           scores = json.loads(resp.read().decode("utf-8"))["scores"]
       return chunk[["id"]].assign(score=scores)

Replace the placeholders with endpoints and a network-link name already
configured for the MaxCompute project. DPE checks these settings before
submitting the UDF task.

If multiple ``with_network_options`` decorators are applied, their public
whitelists are combined, as are their internal whitelists. A UDF can use only
one VPC network link.

.. _udf_config_image:

.. _with-image-options:

Custom images
~~~~~~~~~~~~~

Use ``with_image_options`` to run a UDF in a custom runtime image that has
been registered in MaxCompute. Custom UDF images require DPE;
see :ref:`udf_config_engine_support`.

.. code-block:: python

   from maxframe.udf import with_image_options, with_running_options


   @with_running_options(engine="DPE")
   @with_image_options(image_name="registered_image_name")
   def process_in_custom_image(chunk):
       import custom_scoring_pkg  # Preinstalled in the custom image.

       scores = custom_scoring_pkg.score_batch(chunk["feature"].tolist())
       return chunk[["id"]].assign(score=scores)

Combine decorators
~~~~~~~~~~~~~~~~~~

Multiple decorators can be combined on one UDF. For example, the following UDF
runs with the specified task resources, Pillow, and the ``labels.txt`` resource file
available:

.. code-block:: python

   from io import BytesIO

   from maxframe.udf import with_python_requirements, with_resources, with_running_options


   @with_running_options(cpu=2, memory="8GiB")
   @with_python_requirements("Pillow==10.4.0")
   @with_resources("labels.txt")
   def process_batch(chunk):
       from PIL import Image

       with open("labels.txt", encoding="utf-8") as f:
           default_label = f.readline().strip()
       widths = [
           Image.open(BytesIO(raw)).size[0] for raw in chunk["image_bytes"]
       ]
       return chunk[["id"]].assign(width=widths, label=default_label)

.. _udf_config_defaults:

Set shared defaults with ``options``
------------------------------------

Import ``options`` and set shared values before creating the session or the
MaxFrame expressions that use them:

.. code-block:: python

   from maxframe.config import options

See :ref:`udf_shared_options_reference` for the option names, types, and defaults.
The examples below show common settings.

.. _udf_config_python:

Python runtime
~~~~~~~~~~~~~~

.. code-block:: python

   options.python_tag = "cp311"

``cp311`` selects CPython 3.11; ``cp37`` selects CPython 3.7 in runtimes
that provide it. These are the Python tags documented for the
`built-in images <https://help.aliyun.com/en/maxcompute/built-in-images>`_.
The SDK accepts a string, but the selected runtime must provide that ABI.
Check the image and dependency wheels before changing it.

.. _udf_config_default_resources:

Default task resources
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   options.function.default_running_options = {
       "cpu": 2,
       "memory": "8GiB",
   }

On DPE, each function task requests two CPUs and 8 GiB of memory by default.
On MCSQL, these defaults configure UDF worker resources and produce a CU
setting of 2. The ``memory_limit`` and ``gpu`` fields apply only to DPE.

The shared GPU field is named ``gpu``; the decorator parameter is ``gu``.
For elastic-memory defaults, see :ref:`user_guide_elastic_memory`.

.. _udf_config_quota_image:

Default GPU quota and custom image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   options.session.gu_quota_name = "gpu_quota_name"
   options.session.image_name = "registered_image_name"

Both defaults apply to DPE UDFs. ``gu_quota_name`` is used when a function
requests a GPU without setting ``gu_quota``. ``image_name`` is used when a
function does not set ``with_image_options``; it does not select DPE
automatically. MCSQL does not support either setting.

DPE network whitelists
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   options.dpe.settings["substep.public_network_whitelist"] = [
       "<approved-public-endpoint>",
   ]
   options.dpe.settings["substep.internal_network_whitelist"] = [
       "<approved-internal-endpoint>",
   ]

Replace the placeholders with endpoints configured for the MaxCompute project.
DPE validates both lists before submitting function tasks. A VPC network link
applies to one function and is configured with
``with_network_options(vpc_network_link=...)``.

.. _udf_config_precedence:

Override a shared default for one function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maxframe.udf import with_running_options

   @with_running_options(cpu=4)
   def use_more_cpu(chunk):
       return chunk[["id"]].assign(score=chunk["value"] * 0.2)

With the shared defaults in :ref:`udf_config_default_resources`,
``use_more_cpu`` uses four CPUs and inherits 8 GiB of memory.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Rule
   * - Execution engine
     - ``engine`` explicitly selects an available engine for that function,
       taking precedence over automatic engine selection. Without it,
       MaxFrame follows the job's engine settings.
   * - CPU, memory, memory limit, or GPU
     - A field set by ``with_running_options`` replaces the corresponding
       field in ``default_running_options``. Other fields are inherited.
   * - GPU quota
     - ``gu_quota`` replaces ``options.session.gu_quota_name`` for that
       function.
   * - Custom image
     - For DPE UDFs, ``with_image_options`` replaces
       ``options.session.image_name`` for that function. Neither setting
       selects DPE automatically.
   * - Public or internal network whitelist
     - In DPE, function-level and shared whitelists are combined separately
       for public and internal addresses.

Python dependencies, MaxCompute resource files, OSS mounts, and VPC network
links have no shared ``options`` setting and are configured with decorators.

.. _udf_config_troubleshooting:

Troubleshoot configuration problems
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Problem
     - What to check
   * - An ``options`` setting does not take effect
     - Check that the setting is specified before creating the session or the
       relevant MaxFrame expressions. Then check whether a decorator overrides the same setting.
       See :ref:`udf_config_precedence`.
   * - The task requests more CPUs than configured
     - Check the warning and ``options.function.allowed_max_memory_cpu_ratio``.
       MaxFrame increases the CPU request if the memory per CPU exceeds this
       limit. CPU allocation may also increase during an elastic-memory retry.
       See :ref:`udf_config_resources`.
   * - An elastic-memory configuration fails or has no effect
     - Check :ref:`udf_config_elastic_overrides` for inherited limits and
       :ref:`udf_elastic_memory_not_expanding` for engine and retry restrictions.
   * - ``memory must be specified when memory_limit is set``
     - The merged settings have a limit but no starting memory. Set ``memory``
       in the decorator or in ``options.function.default_running_options``.
       See :ref:`udf_config_elastic_overrides`.
   * - ``gu or gu_type can't be specified with cpu or memory``
     - A GPU request is combined with a CPU or memory request. Remove ``cpu``
       and ``memory`` from the GPU decorator, or remove ``gu`` and ``gu_type``
       to run a CPU task.
   * - ``gu_type requires gu_quota to be specified``
     - Set ``gu_quota`` in the same decorator as ``gu_type``. The shared quota
       default does not satisfy this decorator check. See
       :func:`maxframe.udf.with_running_options`.
   * - A GPU quota is configured, but the task does not request a GPU
     - Also specify a GPU count: use ``gu`` in ``with_running_options``, or ``gpu``
       in ``options.function.default_running_options``. A quota name alone
       does not request GPU resources.
   * - The UDF cannot import a package installed on the client
     - Check whether the remote environment contains the package. Add missing
       packages with ``with_python_requirements``, or use a registered image
       that contains them. See :ref:`udf_config_packages`.
   * - Network configuration validation fails
     - Check that the addresses and network-link name are configured for the
       current MaxCompute project. Public addresses need to be added to the
       whitelist through a service ticket before use.
       See :ref:`udf_config_network`.
