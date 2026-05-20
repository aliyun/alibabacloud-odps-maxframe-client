.. _examples_openlake_multimodal_maxframe:

Multimodal image pipeline with OpenLake, Object Tables, and MaxFrame
====================================================================

For multimodal analytics, **unstructured data at scale** needs both catalog metadata and a distributed runtime. **MaxCompute Object tables** capture OSS object metadata automatically, while **MaxFrame** runs Python transforms on that catalog. Combined with **OpenLake** (DLF + **Paimon**), you can resize images in parallel and land curated bytes in a lake table for retrieval, **AI Functions**, or downstream training.

This walkthrough uses a **public image folder** on OSS: metadata is registered in an object table, a small **UDF** resizes each image to 150×150 BMP, and results are written to **Paimon** through a DLF-backed catalog.

.. only:: html

   .. raw:: html

      <div class="mf-dw-notebook-launch">
        <a class="mf-dw-notebook-btn" href="https://dataworks.console.aliyun.com/gallery/YWxpeXVuL2RhdGFXb3Jrcy9tdWx0aW1vZGFsRGF0YVByb2Nlc3NpbmcudGFyLmd6?templateType=NOTEBOOK&amp;templateFileUrl=https%3A%2F%2Fdsw-js.oss-cn-beijing.aliyuncs.com%2Fproduction%2Fpai-dsw-examples%2Fv0.6.219%2Faliyun%2FdataWorks%2FmultimodalDataProcessing%2Ftgz%2FmultimodalDataProcessing.tar.gz&amp;templateFileVpcUrl=https%3A%2F%2Fdsw-js-cn-hangzhou.oss-cn-hangzhou-internal.aliyuncs.com%2Fproduction%2Fpai-dsw-examples%2Fv0.6.219%2Faliyun%2FdataWorks%2FmultimodalDataProcessing%2Ftgz%2FmultimodalDataProcessing.tar.gz&amp;previewFileUrl=https%3A%2F%2Fdsw-js.data.aliyun.com%2Fproduction%2Fpai-dsw-examples%2Fv0.6.219%2Faliyun%2FdataWorks%2FmultimodalDataProcessing%2Fpreview%2FmultimodalDataProcessing.html&amp;ipynbFileUrl=https%3A%2F%2Fdsw-js.data.aliyun.com%2Fproduction%2Fpai-dsw-examples%2Fv0.6.219%2Faliyun%2FdataWorks%2FmultimodalDataProcessing%2Fipynb%2FmultimodalDataProcessing.ipynb" target="_blank" rel="noopener noreferrer" aria-label="Run this tutorial on DataWorks Notebook">
          <img class="mf-dw-notebook-btn__icon" src="../_static/dataworks-notebook-icon.svg" width="20" height="20" alt="" decoding="async" />
          <span class="mf-dw-notebook-btn__label">Run this tutorial on DataWorks Notebook</span>
        </a>
      </div>

End-to-end flow
---------------

.. code-block:: text

   OSS images  →  Object table (metadata)  →  MaxFrame SQL / DataFrame
        →  UDF resize (distributed apply)
        →  Paimon on DLF (OpenLake)
        →  optional read-back for QA / retrieval prep

Prerequisites
-------------

MaxCompute
^^^^^^^^^^

- A **MaxCompute project** with the **three-layer model** enabled (project / **schema** / object). See `Schemas <https://www.alibabacloud.com/help/en/maxcompute/user-guide/schemas-1>`__.
- **Endpoints** for the API and Tunnel that match your network (VPC ``*.aliyun-inc.com`` or public ``*.aliyun.com``). See `Obtain endpoints <https://www.alibabacloud.com/help/en/maxcompute/user-guide/endpoints>`__.
- **RAM or default credential chain** usable from notebooks (the sample uses ``CredentialProviderAccount`` with ``DefaultCredentialsProvider``).
- Optional **external project** name if you read the Paimon table through a MaxCompute **external project** (``external_project_name`` in the sample).

OpenLake: DLF + Paimon
^^^^^^^^^^^^^^^^^^^^^^^

- A **DLF catalog** in the same region you configure, with **AccessKey** pair or another supported credential for catalog APIs. See `Data Lake Formation <https://www.alibabacloud.com/help/en/dlf/>`__.
- **Paimon** Java/Python client libraries installed in the environment where you run the write step (for example ``paimon_python_java``, ``paimon_python_api``, ``pyarrow``). Package names and install steps follow your OpenLake onboarding guide.

Configuration placeholders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   project_name = "[your-project-name]"
   external_project_name = "[your-external-project-name]"
   region = "cn-shanghai"
   table_schema = "[your-schema]"
   object_table_name = "[your-object-table]"
   paimon_table_name = "[your-paimon-table]"

   endpoint = f"http://service.{region}.maxcompute.aliyun-inc.com/api"
   tunnel_endpoint = f"http://dt.{region}.maxcompute.aliyun-inc.com"

   dlf_region = "[dlf-region]"
   dlf_catalog_id = "[dlf-catalog-id]"
   dlf_database_name = "default"
   dlf_catalog_access_key_id = "[dlf-access-key-id]"
   dlf_catalog_access_key_secret = "[dlf-access-key-secret]"

   dlf_endpoint = f"dlfnext-vpc.{dlf_region}.aliyuncs.com"

Step 1. Preview a public OSS object (optional)
------------------------------------------------

The sample reads a single JPEG from a **public** demo prefix using **anonymous** OSS auth. Replace ``bucket_name`` and ``object_key`` with your own layout when you move to private data.

.. code-block:: python

   import io

   import matplotlib.pyplot as plt
   import oss2
   from PIL import Image

   bucket_name = f"dataworks-notebook-{region}"
   object_key = "public-datasets/L1_Multimodal/cats-vs-sheeps/cat.1.jpg"

   bucket = oss2.Bucket(
       oss2.AnonymousAuth(),
       f"oss-{region}-internal.aliyuncs.com",
       bucket_name,
   )

   image_data = bucket.get_object(object_key).read()
   image = plt.imread(io.BytesIO(image_data), format="jpeg")
   plt.imshow(image)
   plt.axis("off")
   plt.show()

   meta = bucket.head_object(object_key)
   content_length = meta.headers.get("Content-Length")
   print(f"Original size (bytes): {content_length}")

   image = Image.open(io.BytesIO(image_data))
   width, height = image.size
   print(f"Original width: {width}px, height: {height}px")

Step 2. Open a MaxFrame session
-------------------------------

Session flags turn on **schema mode**, tune **object-table splitting** for concurrency, and disable features that conflict with this lab-style job.

.. code-block:: python

   from alibabacloud_credentials import providers
   from maxframe import new_session, options
   import maxframe.dataframe as md
   from odps import ODPS
   from odps.accounts import CredentialProviderAccount

   options.sql.settings = {
       "odps.namespace.schema": "true",
       "odps.task.major.version": "default",
       "odps.sql.allow.namespace.schema": "true",
       "odps.sql.auto.merge.enabled": "false",
       "odps.sql.object.table.split.by.object.size.enabled": "true",
       "odps.sql.object.table.split.unit.kb": "1000",
       "odps.sql.offline.result.cache.enable": "false",
       "odps.sql.split.v2": "false",
       "odps.stage.mapper.split.size": "10",
       "odps.sql.type.system.odps2": "true",
   }

   options.sql.enable_mcqa = False
   options.sql.auto_use_common_image = False
   options.session.enable_schema = True

   account = CredentialProviderAccount(providers.DefaultCredentialsProvider())
   o = ODPS(
       account=account,
       project=project_name,
       endpoint=endpoint,
       tunnel_endpoint=tunnel_endpoint,
   )

   session = new_session(o)
   print(f"MaxFrame session id: {session.session_id}")
   print(session.get_logview_address())

Step 3. Create and refresh the object table
-------------------------------------------

Point ``LOCATION`` at the OSS prefix that stores images. **Refresh metadata** before querying keys from MaxFrame.

.. code-block:: python

   oss_prefix = (
       "oss://oss-cn-shanghai-internal.aliyuncs.com/"
       "dataworks-dataset-cn-shanghai/public-datasets/L1_Multimodal/cats-vs-sheeps/"
   )
   fq_ot = f"{project_name}.{table_schema}.{object_table_name}"

   o.execute_sql(
       f"CREATE OBJECT TABLE IF NOT EXISTS {fq_ot} LOCATION '{oss_prefix}'",
       hints=options.sql.settings,
   )
   o.execute_sql(
       f"ALTER TABLE {fq_ot} REFRESH METADATA",
       hints=options.sql.settings,
   )

Step 4. Inspect object metadata and one image payload
------------------------------------------------------

.. code-block:: python

   ot_sample = (
       md.read_odps_query(f"SELECT key, size, type, owner_id FROM {fq_ot}")
       .execute()
       .fetch()
   )
   print(ot_sample.head(12))

   df = md.read_odps_query(
       f"SELECT key, "
       f"base64(get_data_from_oss('{fq_ot}', key)) AS data "
       f"FROM {fq_ot} WHERE key = 'cat.1.jpg'",
       index_col="key",
   )
   print(df.execute().fetch())

``get_data_from_oss`` is a MaxCompute SQL helper that reads object bytes for a given key; the notebook encodes them as **base64** so the Python UDF can decode without mounting OSS inside the mapper.

Step 5. Resize images in a MaxFrame UDF
---------------------------------------

``@with_python_requirements`` ships **Pillow**, **pandas**, and **cloudpickle** to workers.

.. code-block:: python

   from maxframe.udf import with_python_requirements


   @with_python_requirements("pillow", "pandas", "cloudpickle")
   def apply_func(row):
       import base64
       import io

       from PIL import Image

       src_image = Image.open(io.BytesIO(base64.b64decode(row.iloc[-1])))
       canvas = Image.new(src_image.mode, (150, 150), (0, 0, 0))
       scale = 150.0 / max(src_image.size)
       resized = src_image.resize(
           tuple(int(s * scale) for s in src_image.size)
       )
       canvas.paste(resized, (0, 0))
       sink = io.BytesIO()
       canvas.save(sink, "bmp")
       row = row.copy()
       row.iloc[-1] = base64.b64encode(sink.getvalue()).decode()
       return row

Step 6. Run ``apply`` on MaxCompute
-----------------------------------

.. code-block:: python

   apply_df = df.apply(
       apply_func,
       axis=1,
       dtypes=df.dtypes,
       output_type="dataframe",
   )
   print(apply_df.execute().fetch())

Step 7. Write the result to Paimon (DLF catalog)
------------------------------------------------

Install the **Paimon** Python bindings and **PyArrow** in the client environment that runs this cell. The catalog uses the **``dlf-paimon``** metastore type.

.. code-block:: python

   import pyarrow as pa
   from paimon_python_api import Schema
   from paimon_python_java import Catalog

   catalog_options = {
       "metastore": "dlf-paimon",
       "dlf.endpoint": dlf_endpoint,
       "dlf.region": dlf_region,
       "dlf.catalog.id": dlf_catalog_id,
       "dlf.catalog.accessKeyId": dlf_catalog_access_key_id,
       "dlf.catalog.accessKeySecret": dlf_catalog_access_key_secret,
   }
   catalog = Catalog.create(catalog_options)

   pandas_df = apply_df.to_pandas()
   record_batch = pa.RecordBatch.from_pandas(pandas_df)
   schema = Schema(record_batch.schema)

   fq_paimon = f"{dlf_database_name}.{paimon_table_name}"
   catalog.create_table(fq_paimon, schema, True)
   table = catalog.get_table(fq_paimon)

   write_builder = table.new_batch_write_builder()
   table_write = write_builder.new_write()
   table_commit = write_builder.new_commit()

   table_write.write_arrow_batch(record_batch)
   commit_messages = table_write.prepare_commit()
   table_commit.commit(commit_messages)

   table_write.close()
   table_commit.close()

Step 8. Read back from MaxCompute and visualize
-----------------------------------------------

The sample reads through an **external project** bound to the lake catalog. Adjust the fully qualified name to match your deployment.

.. code-block:: python

   import base64
   import io

   import matplotlib.pyplot as plt
   from PIL import Image

   sql = (
       f"SELECT data FROM {external_project_name}."
       f"{dlf_database_name}.{paimon_table_name}"
   )
   with o.execute_sql(sql, hints=options.sql.settings).open_reader(
       tunnel=False
   ) as reader:
       rec = next(reader)

   buf = io.BytesIO(base64.b64decode(rec[-1]))
   img = Image.open(buf)
   plt.imshow(img)
   plt.show()

   buf.seek(0)
   image = Image.open(buf)
   width, height = image.size
   raw_bytes = base64.b64decode(rec[-1])
   print(f"Result size (bytes): {len(raw_bytes)}")
   print(f"Result width: {width}px, height: {height}px")

Step 9. Close the session
-------------------------

.. code-block:: python

   session.destroy()
   print("Session closed")

Further reading
---------------

- `View and use custom images <https://www.alibabacloud.com/help/en/maxcompute/user-guide/custom-image>`__ if you extend the UDF with heavier dependencies.
- `Schemas and namespace mode <https://www.alibabacloud.com/help/en/maxcompute/user-guide/schemas-1>`__ for three-layer project layout.

Use the **Run this tutorial on DataWorks Notebook** button at the top for the gallery entry that bundles the notebook and template assets.
