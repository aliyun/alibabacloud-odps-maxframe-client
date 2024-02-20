MaxCompute MaxFrame Client
==========================

MaxFrame is a computational framework created by Alibaba Cloud to
provide a way for Python developers to parallelize their code with
MaxCompute. It creates a runnable computation graph locally, submits it
to MaxCompute to execute and obtains results from MaxCompute.

MaxFrame client is the client of MaxFrame. Currently it provides a
DataFrame-based SDK with compatible APIs for pandas. In future, other
common Python libraries like numpy and scikit-learn will be added as
well. Python 3.7 is recommended for MaxFrame client to enable all
functionalities while supports for higher Python versions are on the
way.

Installation
------------

You may install MaxFrame client through PIP:

.. code:: bash

   pip install maxframe

Latest beta version can be installed with ``--pre`` argument:

.. code:: bash

   pip install --pre maxframe

You can also install MaxFrame client from source code:

.. code:: bash

   pip install git+https://github.com/aliyun/alibabacloud-odps-maxframe-client.git

Getting started
---------------

We show a simple code example of MaxFrame client which read data from a
MaxCompute table, performs some simple data transform and writes back
into MaxCompute.

.. code:: python

   import maxframe.dataframe as md
   import os
   from maxframe import new_session
   from odps import ODPS

   o = ODPS(
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
       project='your-default-project',
       endpoint='your-end-point',
   )
   session = new_session(o)

   df = md.read_odps_table("source_table")
   df["A"] = "prefix_" + df["A"]
   md.to_odps_table(df, "prefix_source_table")

Documentation
-------------

Detailed documentations can be found
`here <https://maxframe.readthedocs.io>`__.

License
-------

Licensed under the `Apache License
2.0 <https://www.apache.org/licenses/LICENSE-2.0.html>`__.
