.. _udf_api_reference:

UDF configuration API
=====================

For configuration choices and engine support, see :ref:`udf_configuration`.
This reference lists decorator signatures, parameter defaults, and errors.

.. currentmodule:: maxframe.udf

.. autofunction:: with_running_options

.. autofunction:: with_python_requirements

.. autofunction:: with_resources

.. autofunction:: with_resource_libraries

.. autofunction:: with_fs_mount

.. autofunction:: with_network_options

.. autofunction:: with_image_options

.. _udf_shared_options_reference:

Shared options
--------------

Set these options before creating the session and relevant expressions.
See :ref:`udf_config_precedence` for how decorators interact with defaults.

.. list-table::
   :header-rows: 1
   :widths: 34 18 18 30

   * - Configuration
     - Type
     - Default
     - Meaning
   * - ``options.python_tag``
     - ``str``
     - Detected from the client interpreter
     - Python ABI used by the remote runtime, for example ``"cp311"`` for
       CPython 3.11.
   * - ``options.function.default_running_options``
     - ``dict``
     - ``{"cpu": 1, "memory": "4GiB", "gpu": 0}``
     - CPU, starting memory, and GPU defaults for each function task. An
       optional ``memory_limit`` sets the DPE CPU UDF elastic-memory limit.
   * - ``options.function.allowed_max_memory_cpu_ratio``
     - Positive ``int`` up to 12
     - ``12``
     - Maximum GiB of memory per CPU. MaxFrame increases the CPU request when
       a task exceeds this ratio.
   * - ``options.session.gu_quota_name``
     - ``str`` or ``None``
     - ``None``
     - Default GPU quota nickname. Setting it does not request a GPU.
   * - ``options.session.image_name``
     - ``str`` or ``None``
     - ``None``
     - Default registered MaxCompute image for DPE function tasks. Does not
       select DPE automatically.
   * - ``options.dpe.settings["substep.public_network_whitelist"]``
     - ``list[str]``
     - Not set
     - Public IP addresses or domains requested for DPE function tasks. Submit
       a service ticket before use to have the entries added to the whitelist.
   * - ``options.dpe.settings["substep.internal_network_whitelist"]``
     - ``list[str]``
     - Not set
     - Alibaba Cloud internal IP addresses or domains requested for DPE
       function tasks.
