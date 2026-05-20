.. _skills_index:

Coding Skills
=============

AI-powered programming assistant for distributed data development.

Overview
********

MaxFrame Coding Skill is an AI coding assistant released by Alibaba Cloud
MaxFrame. It integrates with mainstream AI coding assistants as an intelligent
plugin and injects MaxFrame's distributed data processing knowledge into AI
agents, enabling them to generate runnable MaxFrame code from natural language
requirements.

MaxFrame Coding Skill covers the full MaxFrame development workflow, from
session management, data reading and writing, and operator selection to result
writing. It lowers the entry barrier for distributed data processing and
improves coding efficiency.

Architecture
************

MaxFrame Coding Skill uses a multi-layer knowledge injection architecture to
systematically inject the complete development knowledge base into AI agents:

.. code-block:: text

   +---------------------------------------------------+
   |               AI Coding Assistants                |
   |    (Claude Code / Cursor / Codex / Gemini CLI /   |
   |        Tongyi Lingma / OpenCode / ...)            |
   +---------------------------------------------------+
   |            MaxFrame Coding Skill                  |
   |  +----------+  +----------+  +----------+         |
   |  | Coding   |  | Context  |  | Operator |         |
   |  | Skill    |  | Guide    |  | Selector |         |
   |  +----------+  +----------+  +----------+         |
   |  +----------+  +----------+  +----------+         |
   |  | Selection|  | API Docs |  | Operator |         |
   |  | Rules    |  | 900+ pp. |  | Validator|         |
   |  +----------+  +----------+  +----------+         |
   |  +----------------------------------------+       |
   |  |      Production-grade code examples    |       |
   |  +----------------------------------------+       |
   +---------------------------------------------------+
   |               MaxFrame SDK                        |
   |    DataFrame | Tensor | Learn | UDF | Session     |
   +---------------------------------------------------+
   |            MaxCompute distributed engine          |
   +---------------------------------------------------+

.. list-table::
   :header-rows: 1
   :widths: 28 72
   :class: mf-compare-table

   * - Component
     - Capability
   * - Coding skill definition
     - Defines the Skill's core responsibilities, capability boundaries, and workflow.
   * - Context guide
     - A comprehensive 1700+ line reference covering all features from basics to advanced usage.
   * - Operator selector agent
     - An intelligent agent responsible for operator discovery, validation, and recommendation.
   * - Selection rule engine
     - Selection strategies based on performance-first, batch-first, and compatibility-first principles.
   * - API documentation library
     - 900+ pages of complete MaxFrame API documentation with real-time lookup support.
   * - Operator validation scripts
     - Executable scripts that verify whether operators exist and retrieve detailed documentation.
   * - Production examples
     - 10 complete production-grade code templates covering typical scenarios.

Supported Platforms
*******************

MaxFrame Coding Skill supports mainstream AI coding assistants with a unified
installation pattern:

.. list-table::
   :header-rows: 1
   :widths: 40 60
   :class: mf-compare-table

   * - AI coding platform
     - Installation directory
   * - Claude Code
     - ``.claude/skills/``
   * - Cursor
     - ``.cursor/rules/``
   * - Codex
     - ``.codex/skills/``
   * - OpenCode
     - ``.opencode/skills/``
   * - Gemini CLI
     - ``.gemini/skills/``
   * - Tongyi Lingma / Qoder
     - ``.aone_copilot/skills/`` or ``.qoder/skills/``

Installation
************

1. **Download the package**

   Skill package: `maxframe-coding-skill.zip <https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20260424/mqanrl/maxframe-coding-skill.zip>`__

2. **Extract it to the skills directory of your AI coding assistant**. For Claude Code:

   .. code-block:: bash

      unzip maxframe-coding-skill.zip -d your-project/.claude/skills/

3. **Verify the installation**

   .. code-block:: bash

      ls your-project/.claude/skills/maxframe-job-coding/

   The directory should contain ``SKILL.md``, ``examples/``, ``references/``, and ``scripts/``.

4. **After installation, enter the following prompt in your AI coding assistant**

   .. code-block:: text

      Create a MaxFrame job that reads data from the user_behavior table, groups by city to calculate GMV, and writes the result to the city_gmv_report table.

   The AI assistant will automatically:

   - Confirm the data source and output target.
   - Recommend the best operator combination, such as ``groupby().agg()``.
   - Generate runnable code with complete Session management and error handling.

Core Capabilities
*****************

Intelligent Operator Recommendation
-----------------------------------

MaxFrame provides a multi-layer operator system, including standard
pandas-compatible operators, MaxFrame-specific ``.mf`` extension operators
such as ``apply_chunk``, ``map_reduce``, ``flatmap``, and ``rebalance``, and
UDF / UDTF capabilities. For a specific data processing requirement, the
Operator Selector agent built into Coding Skill automatically completes
operator selection and validation:

- **Task-driven recommendation**: recommends the best operator combination based on the task description and explains the reason.
- **API authenticity validation**: validates operators against 900+ pages of API documentation to prevent hallucinated APIs.
- **Fallback alternatives**: provides alternatives, including UDF fallback options, when the preferred operator has constraints.

Example:

.. code-block:: text

   User: "I need a rolling average for time-series data."
   AI:   "Use DataFrame.rolling().
          If you need custom window logic, use .mf.apply_chunk() as an alternative."

End-to-End Code Generation
--------------------------

Coding Skill covers the complete MaxFrame development lifecycle:

.. code-block:: text

   Session creation -> data reading -> operator selection -> data processing -> result writing -> Session cleanup

It uses a three-phase confirm-before-execute interaction model to ensure the
generated code precisely matches the requirement:

.. list-table::
   :header-rows: 1
   :widths: 22 32 46
   :class: mf-compare-table

   * - Phase
     - Content
     - Description
   * - Phase 1
     - Requirement and data confirmation
     - Confirms data sources, target tables, selected columns, and related inputs.
   * - Phase 2
     - Operator selection confirmation
     - Shows recommended operators and alternatives, then waits for confirmation.
   * - Phase 3
     - Code generation and validation
     - Generates complete runnable code based on the confirmed plan.

All generated code follows production-grade standards:

- Uses ``try/finally`` to ensure Session resource cleanup.
- Automatically calls ``.execute()`` to trigger lazy execution.
- Correctly declares UDF return types with ``dtypes``.
- Includes complete error handling logic.

Common Pitfall Prevention
-------------------------

Generic AI-generated MaxFrame code often runs into the following issues. Coding
Skill solves them with its built-in knowledge base:

.. list-table::
   :header-rows: 1
   :widths: 36 64
   :class: mf-compare-table

   * - Common issue
     - How Coding Skill solves it
   * - Calling nonexistent APIs
     - Validates against 900+ pages of documentation to prevent hallucinated APIs.
   * - Missing ``.execute()`` calls
     - Enforces lazy execution patterns and includes execution triggers in code templates.
   * - Session not destroyed correctly
     - Uses ``try/finally`` in all generated code to release resources.
   * - UDF return type mismatch
     - Shows the correct ``dtypes`` declaration pattern through examples.
   * - Poor execution engine choice
     - Recommends engines by SQL Engine > DPE > SPE priority.
   * - Inefficient operators
     - Recommends ``DataFrame.mf.apply_chunk`` instead of ``Series.apply`` where appropriate.

Built-In Scenario Templates
---------------------------

Coding Skill includes 10 production-grade code templates for typical business
scenarios. AI agents can use these templates to generate high-quality code:

.. list-table::
   :header-rows: 1
   :widths: 24 34 42
   :class: mf-compare-table

   * - Scenario
     - Example file
     - Core capability
   * - LLM batch inference
     - ``ai_function_basic.py``
     - Distributed batch inference with ManagedTextLLM, ready to use out of the box.
   * - GPU-accelerated computing
     - ``gpu_unit_dpe_processing.py``
     - GPU resource allocation with ``@with_running_options(gu=1)``.
   * - OSS file processing
     - ``fs_mount_example.py``
     - Distributed OSS file reading with ``@with_fs_mount``.
   * - Multiple OSS mounts
     - ``oss_multi_mount.py``
     - Mounts one or more OSS buckets at the same time.
   * - Grouped batch processing
     - ``groupby_batch_processing.py``
     - Efficient grouped batch processing with ``groupby`` + ``apply_chunk``.
   * - Complex data structures
     - ``complex_struct.py``
     - Nested structures with custom grouped processing.
   * - Arrow type handling
     - ``complex_struct_arrow.py``
     - PyArrow complex types with JSON conversion.
   * - DLF external table writes
     - ``dlf_table_write_basic.py``
     - DLF external table configuration and data writing.
   * - DLF primary-key table writes
     - ``dlf_table_write_with_pk.py``
     - Primary-key tables with binary data type handling.
   * - Large-scale document deduplication
     - ``minhash_lsh_document_similarity.py``
     - MinHash + LSH algorithm with 4000+ parallelism support.

Typical Scenarios
*****************

Scenario 1: Distributed Batch LLM Inference
-------------------------------------------

Requirement: run batch inference on massive text data with a large language
model.

No model deployment, GPU resource management, or inference service development
is required. ManagedTextLLM provides built-in qwen2.5 series models,
DeepSeek-R1, and more.

Generated code example:

.. code-block:: python

   import os

   import maxframe.dataframe as md
   from maxframe.learn.contrib.llm.models.managed import ManagedTextLLM
   from maxframe.session import new_session
   from odps import ODPS

   o = ODPS(
       os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID"),
       os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
       project="your-default-project",
       endpoint="your-end-point",
   )

   session = new_session(o)

   try:
       df = md.DataFrame(
           {
               "query": [
                   "What is the average distance from Earth to the Sun?",
                   "What is the boiling point of water?",
               ]
           }
       )
       df.execute()

       llm = ManagedTextLLM(name="qwen2.5-1.5b-instruct")
       messages = [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "{query}"},
       ]
       result = llm.generate(df, prompt_template=messages)
       result.execute()
   finally:
       session.destroy()

Scenario 2: Distributed OSS File Processing
-------------------------------------------

Requirement: mount files from OSS to every distributed Worker node for parallel
reading and processing.

OSS paths are mounted as local file system paths. Distributed Workers read data
in parallel, and throughput scales with the number of nodes.

Generated code example:

.. code-block:: python

   from maxframe.udf import with_fs_mount, with_running_options


   @with_running_options(engine="dpe", cpu=2, memory=4)
   @with_fs_mount(
       "oss://your-bucket/model-files/",
       "/mnt/model",
       storage_options={"role_arn": "acs:ram::xxx:role/xxx"},
   )
   def read_model_directory(row):
       import os

       files = os.listdir("/mnt/model")
       # Each Worker reads independently in distributed parallel mode.
       ...

Related Links
*************

- Skill repository: `aliyun/alibabacloud-aiops-skills <https://github.com/aliyun/alibabacloud-aiops-skills/tree/master/skills/analyticscomputing/odps/alibabacloud-odps-maxframe-coding>`__
- MaxFrame official documentation: `MaxFrame distributed AI computing engine <https://help.aliyun.com/zh/maxcompute/user-guide/maxframe-overview-1/>`__
- `MaxFrame FAQ <https://help.aliyun.com/zh/maxcompute/user-guide/faq-about-maxframe>`__
