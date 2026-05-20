.. _user_guide_local_debug:

Local debug mode
----------------

This page introduces the features and usage of MaxFrame Local Debug Mode.
Local Debug Mode allows you to debug UDF functions such as ``apply()`` and
``apply_chunk()`` directly in your local environment, without connecting to
remote services.

Background
~~~~~~~~~~
In the traditional MaxFrame UDF development workflow, debugging functions
such as ``apply()`` and ``apply_chunk()`` requires submitting code to a
remote cluster for execution. This makes it impossible to set breakpoints
or step through code locally. Every change has to be re-submitted for
remote execution, and you may have to maintain different code paths to
distinguish local development and production.

MaxFrame Local Debug Mode solves these problems. Once Local Debug Mode is
enabled, UDF functions are executed directly in the local Python
environment with full IDE breakpoint debugging support, work fully offline,
and the same code can be switched seamlessly between local debugging and
production runs.

Use cases
~~~~~~~~~

.. csv-table::
   :header: "Scenario", "Description"

   "UDF logic development", "Debug and verify complex business logic in real time."
   "Data transformation tests", "Validate data cleaning and transformation rules."
   "Issue investigation", "Locate the root cause of UDF execution failures."
   "Offline development", "Continue development without network access."

Features
~~~~~~~~

Compared with the traditional remote debugging workflow, Local Debug Mode
provides the following **advantages**:

.. csv-table::
   :header: "Dimension", "Local Debug Mode", "Traditional approach"

   "Breakpoint debugging", "**IDE breakpoints supported**", "Not supported"
   "Remote dependency", "**Fully offline local debugging**", "Requires connection to a remote cluster"
   "Debug cycle", "**Local instant execution**", "Each change must be submitted for remote execution"
   "Code modification", "**A single code base**", "Multiple code paths must be maintained"

* **Zero-configuration debugging**

  Just set ``debug=True`` or ``debug="local"``. No additional tools or
  services are required.

  .. code-block:: python

      session = new_session(o, debug=True)

* **Fully offline**

  Does not depend on networking or remote cluster resources.

* **Native IDE support**

  - Supports mainstream IDEs such as PyCharm and VSCode, as well as
    DataWorks Notebook.
  - Preserves the full debugging experience, including
    **breakpoints, variable watches, and step-by-step execution**.
  - Provides the same debugging experience as local Python development.

* **Flexible data sources**

  Supports in-memory data, local files, MaxCompute tables, and more.

  .. csv-table::
     :header: "Data source", "How to load", "Use case"

     "In-memory data", "``md.DataFrame(pd.DataFrame())``", "Quick logic verification"
     "MaxCompute table", "``md.read_odps_table()``", "Real-data testing"
     "Local file", "``pd.read_csv()`` and other native pandas APIs", "Offline development"

* **Seamless switch to production**

  Debug code is identical to production code. After you remove
  ``debug=True`` or ``debug="local"``, the code can run in production
  directly.

  .. code-block:: python

      # Debugging
      session = new_session(o, debug=True)

      # Production
      session = new_session(o)

Quick start
~~~~~~~~~~~

1. **Prepare the environment**

   .. code-block:: bash

       # MaxFrame SDK 2.5.0 or later is required
       pip install --upgrade maxframe

2. **Basic example**

   .. code-block:: python

       from odps import ODPS
       from maxframe import new_session
       import maxframe.dataframe as md
       import pandas as pd

       o = ODPS(
           access_id='your_access_id',
           secret_access_key='your_secret_key',
           project='your_project',
           endpoint='your_endpoint',
       )

       # Enable debug mode
       session = new_session(o, debug=True)

       df = md.DataFrame(pd.DataFrame({
           "sales": [5000, 8000, 12000, 3000],
           "region": ["A", "B", "C", "D"],
       }))

       def calculate_commission(row):
           sales = row['sales']
           if sales > 10000:  # set a breakpoint here
               rate = 0.15
               print(rate)
           elif sales > 5000:  # set a breakpoint here
               rate = 0.10
               print(rate)
           else:
               rate = 0.05
           return sales * rate

       result = df.apply(calculate_commission, axis=1).execute().fetch()

Notes
~~~~~

* **Performance differences**: Local Debug Mode is intended for development
  and verification. Its performance does not match the production
  environment.
* **Data volume limit**: Use small sample datasets when debugging.
* **Dependency consistency**: Make sure the local Python environment uses
  the same dependency versions as production.
* **Sensitive data**: When debugging MaxCompute tables, follow data
  permission and data masking requirements.
