.. _udf_examples:


UDF examples
============

The DataFrame examples below share these four records. The ``id`` index
identifies the source row in the output, including when one row becomes several
rows. The Tensor example creates its own array.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import maxframe.dataframe as md

   pdf = pd.DataFrame(
       {
           "customer": ["A", "A", "B", "B"],
           "amount": [10.0, 30.0, 20.0, 40.0],
           "text": ["red blue", "blue", "red green", "green blue"],
       },
       index=pd.Index([101, 102, 103, 104], name="id"),
   )
   df = md.DataFrame(pdf)

Each example produces a ``result`` expression. The text block below it shows
the result of ``result.execute().fetch()`` in a configured MaxFrame session.
For connection settings, see :ref:`udf_complete_example`. Result row order may
differ after grouping or repartitioning.

Process individual values
-------------------------

Use this when each value can be handled independently, for example to classify
amounts or convert a field using custom rules. The function below receives
one amount and returns one label.

.. code-block:: python

   def label_amount(value):
       return "large" if value >= 30 else "regular"

   result = df["amount"].map(label_amount, dtype="object")

.. code-block:: text

   id
   101    regular
   102      large
   103    regular
   104      large
   Name: amount, dtype: object

For a Python function over Series values, ``Series.apply`` is another option. Use
``DataFrame.map`` to apply a scalar-to-scalar function to every cell;
``DataFrame.applymap`` is an alias.

Process rows or columns
-----------------------

Use ``DataFrame.apply(axis=1)`` when the calculation needs several fields from
the same record. Here customer A gets a 10 percent discount and customer B
gets a 20 percent discount. The function receives one row as a pandas Series.

.. code-block:: python

   def payable(row):
       rate = 0.9 if row["customer"] == "A" else 0.8
       return row["amount"] * rate

   result = df.apply(
       payable, axis=1, output_type="series", dtype="float64", name="payable"
   )

.. code-block:: text

   id
   101     9.0
   102    27.0
   103    16.0
   104    32.0
   Name: payable, dtype: float64

With ``axis=0``, the function receives one column instead. MCSQL does not
support column-wise ``DataFrame.apply``, so this example explicitly selects
DPE. The function returns the difference between the column's maximum and
minimum values:

.. code-block:: python

   from maxframe.udf import with_running_options


   @with_running_options(engine="DPE")
   def value_range(column):
       return column.max() - column.min()

   result = df[["amount"]].apply(
       value_range, axis=0, output_type="series", dtype="float64"
   )

.. code-block:: text

   amount    30.0
   dtype: float64

For this simple calculation, built-in ``max`` and ``min`` are sufficient;
the example shows the difference between
receiving a row and receiving a column.

Process a batch of rows
-----------------------

Use ``apply_chunk`` when a library or service can process several records at
once, such as batch model inference or vectorized pandas/NumPy operations. The
function receives a pandas DataFrame rather than a single row.

Chunk sizes are not guaranteed to stay the same between executions, even for
the same job. A UDF should handle batches of different sizes and must not
depend on a particular chunk boundary.

.. code-block:: python

   def extract_features(batch: pd.DataFrame) -> pd.DataFrame[
       {"log_amount": "float64", "word_count": "int64"}
   ]:
       log_scores = np.round(np.log1p(batch["amount"].to_numpy()), 2)
       word_counts = batch["text"].str.split().str.len().to_numpy()
       return pd.DataFrame(
           {"log_amount": log_scores, "word_count": word_counts},
           index=batch.index,
       )

   result = df.mf.apply_chunk(
       extract_features,
       batch_rows=2,
       index=df.index,
   )

.. code-block:: text

        log_amount  word_count
   id
   101        2.40           2
   102        3.43           1
   103        3.04           2
   104        3.71           2

Keep the returned pandas index consistent with the MaxFrame declaration;
see :ref:`udf_result_index` for the two ``index`` arguments.

``batch_rows=2`` makes the four-row example run in small batches. It is a
teaching value, not a recommended production batch size. Start with a batch
size that fits the function's memory needs. For ungrouped ``apply_chunk``,
omitting ``batch_rows`` processes each input chunk in one call in DPE; in
MCSQL, the default is 1,024 rows per call. See :ref:`udf_batch_rows`.

``Series.mf.apply_chunk`` works similarly but passes a pandas Series. For return
type declarations and ``skip_infer``, see :ref:`udf_result_types`.

Expand one record into several
------------------------------

Use ``flatmap`` when one input record can produce zero, one, or several output
records. For example, split an article into paragraphs, a video into frames,
or a line of text into words. The function below receives one row and yields
one output row per word.

.. code-block:: python

   def split_words(row):
       for word in row["text"].split():
           yield row["customer"], word

   result = df.mf.flatmap(
       split_words, dtypes={"customer": "object", "word": "object"}
   )

.. code-block:: text

       customer   word
   id
   101        A    red
   101        A   blue
   102        A   blue
   103        B    red
   103        B  green
   104        B  green
   104        B   blue

Each expanded row automatically repeats its source row index (``101, 101,
102...``). For a single text column, ``Series.mf.flatmap`` can receive one text
value directly. ``flatmap`` never infers output types automatically; always
declare ``dtypes`` (or ``dtype`` for scalar Series output).

.. _udf_group_index:

Process a complete group
------------------------

Use ``GroupBy.apply`` when a calculation needs all rows of each group, such as
fitting a model per customer or comparing each order with that customer's
total. This example calculates each order's share of the customer's spending.

.. important::

   Pass ``prepend_index_group_keys=True`` to ``GroupBy.apply`` and grouped
   ``apply_chunk`` to prepend group keys when ``group_keys=True``. Omitting
   this argument uses the current legacy index behavior; specify it explicitly
   for the MultiIndex shown below.

.. code-block:: python

   def order_share(group):
       return group.assign(share=group["amount"] / group["amount"].sum())

   result = df.groupby("customer")[["amount"]].apply(
       order_share,
       output_type="dataframe",
       dtypes={"amount": "float64", "share": "float64"},
       index=df.index,
       prepend_index_group_keys=True,
   )

.. code-block:: text

                amount     share
   customer id
   A        101    10.0  0.250000
            102    30.0  0.750000
   B        103    20.0  0.333333
            104    40.0  0.666667

The function receives the complete group, so its denominator is the customer's
total. ``prepend_index_group_keys=True`` prepends the group key ``customer`` to
the input index ``id`` to form a ``MultiIndex (customer, id)``; ``dtypes``
declares only the columns returned by ``order_share`` itself. For standard
aggregations such as sums and means, use built-in
``agg`` or ``transform`` operations.

Process batches within a group
------------------------------

Use ``DataFrameGroupBy.mf.apply_chunk`` when rows must be grouped by a
business key, but each batch can be processed independently. Each call
contains rows from one group. This example applies a different discount to
each customer.

.. code-block:: python

   def apply_discount(batch):
       customer = batch["customer"].iloc[0]
       rate = {"A": 0.9, "B": 0.8}[customer]
       return pd.DataFrame(
           {"discounted_amount": batch["amount"] * rate},
           index=batch.index,
       )

   result = df.groupby("customer")[["customer", "amount"]].mf.apply_chunk(
       apply_discount,
       batch_rows=1,
       output_type="dataframe",
       dtypes={"discounted_amount": "float64"},
       index=df.index,
       prepend_index_group_keys=True,
   )

.. code-block:: text

                discounted_amount
   customer id
   A        101                9.0
            102               27.0
   B        103               16.0
            104               32.0

Selecting ``customer`` after ``groupby`` makes the group key available inside the
function. Each group has two rows here, and ``batch_rows=1`` splits it into two
calls. This value only demonstrates multiple calls on a small sample; it is
not a recommended production batch size. Results from separate batches are
concatenated, not aggregated into one result per group. Use ``GroupBy.apply``
when the function needs the complete group.
The same :ref:`group-index requirement <udf_group_index>` applies here.

Transform and aggregate by key
------------------------------

Use ``DataFrame.mf.map_reduce`` to produce intermediate records and then
combine records with the same key. This example splits text into words and
counts each word across all rows.

The reducer below accumulates a count over several calls:

* It selects ``engine="MCSQL"`` because this example uses the MCSQL reducer's
  ``end=True`` signal for the last batch of a key.
* With ``end=False``, it adds the current batch to the total and returns an
  empty DataFrame with the declared columns and dtypes.
* With ``end=True``, it returns the total and resets ``self.total`` to zero.
  The same reducer instance can then process another word.

.. code-block:: python

   from maxframe.udf import with_running_options


   def emit_word_rows(batch):
       words = batch["text"].str.split().explode()
       return pd.DataFrame({"word": words, "count": 1})


   @with_running_options(engine="MCSQL")
   class CountWords:
       def __init__(self):
           self.total = 0

       def __call__(self, batch, end=False):
           self.total += int(batch["count"].sum())
           if not end:
               return pd.DataFrame({
                   "word": pd.Series(dtype="str"),
                   "count": pd.Series(dtype="int64"),
               })
           result = pd.DataFrame({
               "word": [batch["word"].iloc[0]],
               "count": [self.total],
           })
           self.total = 0
           return result


   result = df[["text"]].mf.map_reduce(
       emit_word_rows,
       CountWords,
       group_cols=["word"],
       batch_rows=2,
       mapper_dtypes={"word": "str", "count": "int64"},
       mapper_index=pd.Index([], dtype="int64", name="id"),
       reducer_dtypes={"word": "str", "count": "int64"},
       reducer_index=pd.Index([], dtype="int64"),
       ignore_index=True,
   )

.. code-block:: text

       word  count
   0   blue      3
   1  green      2
   2    red      2

Here ``batch_rows=2`` demonstrates batching in both stages. Use a larger value
for larger data, chosen according to memory use. ``mapper_batch_rows`` and
``reducer_batch_rows`` set the two stages separately.
``ignore_index=True`` resets the index of the final ``map_reduce`` result.

``mapper_index`` and ``reducer_index`` declare the index types, not the actual
row labels. The mapper expands each input row into several word rows, so its
index template is an empty integer index named ``id``, rather than
``df.index``.

Process array chunks
--------------------

Use ``Tensor.mf.apply_chunk`` to apply NumPy or other array-processing code to
each Tensor block. The function receives an array block, not a pandas object.
This example clips values into the range from 0 to 1.

.. code-block:: python

   import numpy as np
   import maxframe.tensor as mt


   def clip_values(block):
       return np.clip(block, 0.0, 1.0)

   values = mt.array([[-1.0, 0.2], [0.8, 2.0]], chunk_size=(1, 2))
   result = values.mf.apply_chunk(clip_values, dtype="float64", elementwise=True)

.. code-block:: text

   array([[0. , 0.2],
          [0.8, 1. ]])

Each block is processed independently; a function cannot assume that it
receives the entire Tensor.
``elementwise=True`` declares that each value is processed independently and
the input shape is preserved.
