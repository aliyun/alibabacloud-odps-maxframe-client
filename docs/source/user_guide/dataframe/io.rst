Data input and output
---------------------
.. currentmodule:: maxframe.dataframe

MaxCompute tables
~~~~~~~~~~~~~~~~~
Users can create MaxFrame DataFrame objects from MaxCompute tables with :func:`read_odps_table`
, and store computed results into MaxCompute tables with :func:`DataFrame.to_odps_table`.

For instance, if you want to get data from non-partitioned table `test_odps_table`,
do some transformation by MaxFrame and store it into another partitioned table
`test_processed_odps_table`, you may use :func:`read_odps_table` as is shown below.

.. code-block:: python

    import maxframe.dataframe as md

    df = md.read_odps_table("test_odps_table")
    processed_df = df[df.A > 10]
    processed_df.to_odps_table("test_processed_odps_table")

If the table is partitioned, ``read_odps_table`` will read data from all partitions which
should be definitely avoided if there are a number of partitions. You can select one
partition or a number of partitions with ``partitions`` argument.

.. code-block:: python

    df = md.read_odps_table(
        "parted_odps_table", partitions=["pt1=20240119,pt2=10", "pt1=20240119,pt2=11"]
    )

Values of partition columns are not included in results by default. If you need these values,
you may specify ``append_partitions=True``.

.. code-block:: python

    df = md.read_odps_table(
        "parted_odps_table", partitions=["pt1=20240119,pt2=10"], append_partitions=True
    )

The resulting DataFrame will produce a RangeIndex by default. You may use ``index_col``
argument to specify existing columns as indexes.

.. code-block:: python

    df = md.read_odps_table(
        "parted_odps_table", partitions=["pt1=20240119,pt2=10"], index_col=["idx_col"]
    )

If you want to store prepreocessed ``df`` into a MaxCompute table, you can use :func:`to_odps_table`
as is shown below.

.. code-block:: python

    df.to_odps_table("output_table_name").execute()

You can control the behavior of index output via ``index`` and ``index_label`` arguments.
By default the index is outputted. If you do not want to output the index, you may specify
``index`` argument as False.

.. code-block:: python

    df.to_odps_table("output_table_name", index=False).execute()

The names of columns for indexes is the names of the indexes by default. If names of indexes
are not specified, the name ``index`` will be used if the index only has one level, or
``level_x`` will be used, where ``x`` is the integer index of the level.

Data can be stored as partitioned tables. You may specify ``partition`` argument as the partition
to write.

.. code-block:: python

    df.to_odps_table("parted_table", partition="pt=20240121,h=12").execute()

You can also specify columns as partition columns. The data of these columns will dynamically
decide the partition the row will be written to.

.. code-block:: python

    df.to_odps_table("parted_table", partition_col=["pt_col"]).execute()

pandas objects
~~~~~~~~~~~~~~
Users can convert between local pandas objects and DataFrames with :func:`read_pandas`
and :func:`DataFrame.to_pandas`.

When ``read_pandas`` is called, these pandas objects will be uploaded to MaxCompute and
be used in the cluster.

.. code-block:: python

    md_df = md.read_pandas(pd_df)

After transformation is done in MaxFrame, data can be downloaded to client with ``to_pandas``.

.. code-block:: python

    pd_df = md_df.to_pandas()
