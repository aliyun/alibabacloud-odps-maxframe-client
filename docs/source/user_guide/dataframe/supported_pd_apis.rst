Supported pandas APIs
---------------------
The table below shows implementation of pandas APIs on MaxFrame on certain engines.
If the API is not fully supported, unsupported item will be shown in the detail
column.

Series API
~~~~~~~~~~
.. currentmodule:: maxframe.dataframe.Series

.. list-table::
   :header-rows: 1

   * - API
     - SQL Engine
     - DPE
     - SPE
     - Details
   * - :func:`add`, :func:`radd`
     - P
     - Y
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`add_prefix`
     - Y
     - Y
     - Y
     -
   * - :func:`add_suffix`
     - Y
     - Y
     - Y
     -
   * - :func:`agg`
     - P
     - Y
     - Y
     - SQL engine: customized aggregation not supported.
   * - :func:`align`
     - N
     - Y
     - Y
     -
   * - :func:`all`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`any`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`append`
     - P
     - Y
     - Y
     -
   * - :func:`apply`
     - P
     - Y
     - Y
     -
   * - :func:`argmax`
     - N
     - Y
     - N
     -
   * - :func:`argmin`
     - N
     - Y
     - N
     -
   * - :func:`argsort`
     - N
     - Y
     - N
     -
   * - :func:`astype`
     - P
     - Y
     - Y
     - SQL engine: converting to categorical types not supported.
   * - :func:`autocorr`
     - N
     - P
     - Y
     - DPE engine: only pearson correlation coefficient is supported.
   * - :func:`between`
     - Y
     - Y
     - Y
     -
   * - :func:`case_when`
     - Y
     - N
     - Y
     -
   * - :func:`clip`
     - N
     - Y
     - Y
     -
   * - :func:`compare`
     - N
     - Y
     - Y
     -
   * - :func:`corr`
     - N
     - P
     - Y
     - DPE engine: only pearson correlation coefficient is supported.
   * - :func:`count`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`cov`
     - N
     - Y
     - Y
     -
   * - :func:`cummax`
     - N
     - Y
     - Y
     -
   * - :func:`cummin`
     - N
     - Y
     - Y
     -
   * - :func:`cumprod`
     - N
     - Y
     - Y
     -
   * - :func:`cumsum`
     - N
     - Y
     - Y
     -
   * - :func:`describe`
     - Y
     - P
     - Y
     - DPE engine: medians and percentiles not supported by now.
   * - :func:`diff`
     - N
     - Y
     - Y
     -
   * - :func:`div`, :func:`rdiv`
     - P
     - Y
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`dot`
     - N
     - Y
     - Y
     -
   * - :func:`drop`
     - Y
     - Y
     - Y
     -
   * - :func:`drop_duplicates`
     - P
     - Y
     - Y
     - SQL engine: maintaining original order of data not supported.
   * - :func:`droplevel`
     - N
     - Y
     - Y
     -
   * - :func:`dropna`
     - Y
     - Y
     - Y
     -
   * - :func:`duplicated`
     - N
     - Y
     - Y
     -
   * - :func:`empty`
     - Y
     - Y
     - Y
     -
   * - :func:`eq`, :func:`ne`
     - P
     - Y
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`explode`
     - Y
     - Y
     - Y
     -
   * - :func:`fillna`
     - P
     - Y
     - Y
     - SQL engine: argument ``downcast``, ``limit`` and ``method`` not supported.
   * - :func:`filter`
     - N
     - Y
     - Y
     -
   * - :func:`first_valid_index`
     - N
     - Y
     - Y
     -
   * - :func:`floordiv`, :func:`rfloordiv`
     - P
     - Y
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`ge`, :func:`gt`
     - P
     - Y
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - ``series[item]`` (or ```__getitem__``)
     - N
     - Y
     - Y
     -
   * - :func:`hasnans`
     - N
     - Y
     - Y
     -
   * - :func:`head`
     - Y
     - Y
     - Y
     -
   * - :func:`hist`
     - Y
     - Y
     - Y
     -
   * - :func:`iat`
     - N
     - Y
     - Y
     -
   * - :func:`idxmax`
     - N
     - Y
     - Y
     -
   * - :func:`idxmin`
     - N
     - Y
     - Y
     -
   * - :func:`iloc`
     - P
     - Y
     - Y
     - SQL engine: Non-continuous indexes or negative indexes (for instance, ``df.iloc[[1, 3]]``, ``df.iloc[1:10:2]`` or ``df.iloc[-3:]``) not supported.
   * - :func:`is_monotonic`, :func:`is_monotonic_decreasing`, :func:`is_monotonic_increasing`
     - N
     - Y
     - Y
     -
   * - :func:`is_unique`
     - N
     - Y
     - Y
     -
   * - :func:`isin`
     - P
     - Y
     - Y
     - SQL engine: index input not supported.
   * - :func:`isna`, :func:`notna`
     - Y
     - Y
     - Y
     -
   * - :func:`isnull`, :func:`notnull`
     - Y
     - Y
     - Y
     -
   * - :func:`items`
     - Y
     - Y
     - Y
     -
   * - :func:`kurtosis`
     - N
     - Y
     - Y
     -
   * - :func:`last_valid_index`
     - N
     - Y
     - Y
     -
   * - :func:`le`, :func:`lt`
     - P
     - Y
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`loc`
     - N
     - Y
     - Y
     -
   * - :func:`map`
     - P
     - Y
     - Y
     - SQL engine: argument ``arg`` only supports functions and non-derivative dicts with simple scalars.
   * - :func:`mask`
     - N
     - Y
     - Y
     -
   * - :func:`max`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`mean`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`memory_usage`
     - N
     - Y
     - Y
     -
   * - :func:`min`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`mod`, :func:`rmod`
     - P
     - Y
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`mul`, :func:`rmul`
     - P
     - Y
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`nunique`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``dropna==False`` not supported.
   * - :func:`pct_change`
     - N
     - Y
     - Y
     -
   * - :func:`plot`
     - Y
     - Y
     - Y
     -
   * - :func:`pow`, :func:`rpow`
     - P
     - Y
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`prod`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`quantile`
     - Y
     - Y
     - Y
     -
   * - :func:`reindex`
     - P
     - Y
     - Y
     -
   * - :func:`reindex_like`
     - P
     - Y
     - Y
     -
   * - :func:`rename`
     - P
     - Y
     - Y
     - MCSQL engine: renaming indexes not supported.
   * - :func:`rename_axis`
     - Y
     - Y
     - Y
     -
   * - :func:`reorder_levels`
     - N
     - Y
     - Y
     -
   * - :func:`replace`
     - P
     - Y
     - Y
     - SQL engine: when the argument ``regex`` is True, ``list`` or ``dict`` typed ``to_replace`` not supported. ``list`` or ``dict`` typed ``regex`` argument not supported.
   * - :func:`reset_index`
     - P
     - Y
     - Y
     -
   * - :func:`round`
     - Y
     - Y
     - Y
     -
   * - :func:`sample`
     - P
     - Y
     - Y
     - SQL engine: argument ``replace`` and ``weights`` not supported. ``frac>1`` not supported.
   * - :func:`sem`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`set_axis`
     - N
     - Y
     - Y
     -
   * - ``series[item] = value`` (or ```__setitem__``)
     - P
     - Y
     - Y
     - SQL engine: not supported when ``item`` is callable or DataFrame / Series or index of ``series`` and ``item`` are different.
   * - :func:`shift`
     - N
     - Y
     - Y
     -
   * - :func:`size`
     - P
     - Y
     - Y
     -
   * - :func:`skew`
     - N
     - Y
     - Y
     -
   * - :func:`sort_index`
     - P
     - Y
     - Y
     - SQL engine: ``na_position=='last'`` not supported.
   * - :func:`sort_values`
     - P
     - Y
     - Y
     - SQL engine: ``na_position=='last'`` not supported.
   * - :func:`std`
     - P
     - Y
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`sub`, :func:`rsub`
     - P
     - Y
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`sum`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`swaplevel`
     - N
     - Y
     - Y
     -
   * - :func:`tail`
     - N
     - Y
     - Y
     -
   * - :func:`take`
     - N
     - Y
     - Y
     -
   * - :func:`to_frame`
     - Y
     - Y
     - Y
     -
   * - :func:`transform`
     - P
     - Y
     - Y
     - SQL engine: not supported when ``func`` is dict-like or list-like.
   * - :func:`truediv`, :func:`rtruediv`
     - P
     - Y
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`tshift`
     - N
     - Y
     - Y
     -
   * - :func:`unique`
     - P
     - Y
     - Y
     -
   * - :func:`unstack`
     - N
     - Y
     - Y
     -
   * - :func:`value_counts`
     - Y
     - Y
     - Y
     -
   * - :func:`var`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`where`
     - N
     - Y
     - Y
     -
   * - :func:`xs`
     - N
     - Y
     - Y
     -

DataFrame API
~~~~~~~~~~~~~
.. currentmodule:: maxframe.dataframe.DataFrame

.. list-table::
   :header-rows: 1

   * - API
     - SQL Engine
     - DPE
     - SPE
     - Details
   * - :func:`add`, :func:`radd`
     - P
     - Y
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`add_prefix`
     - Y
     - Y
     - Y
     -
   * - :func:`add_suffix`
     - Y
     - Y
     - Y
     -
   * - :func:`agg`
     - P
     - Y
     - Y
     - SQL engine: customized aggregation not supported.
   * - :func:`all`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`any`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`apply`
     - P
     - Y
     - Y
     - SQL engine: ``axis==0`` not supported. Series output for ``axis==1`` not supported.
   * - :func:`assign`
     - Y
     - Y
     - Y
     -
   * - :func:`astype`
     - P
     - Y
     - Y
     - SQL engine: converting to categorical types not supported.
   * - :func:`at`
     - N
     - Y
     - Y
     -
   * - :func:`axes`
     - Y
     - Y
     - Y
     -
   * - :func:`clip`
     - N
     - Y
     - Y
     -
   * - :func:`compare`
     - N
     - Y
     - Y
     -
   * - :func:`corr`
     - N
     - P
     - Y
     - DPE engine: only pearson correlation coefficient is supported.
   * - :func:`corrwith`
     - N
     - P
     - Y
     - DPE engine: only pearson correlation coefficient is supported.
   * - :func:`count`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`cov`
     - N
     - Y
     - Y
     -
   * - :func:`cummax`
     - N
     - Y
     - Y
     -
   * - :func:`cummin`
     - N
     - Y
     - Y
     -
   * - :func:`cumprod`
     - N
     - Y
     - Y
     -
   * - :func:`cumsum`
     - N
     - Y
     - Y
     -
   * - :func:`describe`
     - Y
     - P
     - Y
     - DPE engine: medians and percentiles not supported by now.
   * - :func:`diff`
     - N
     - Y
     - Y
     -
   * - :func:`div`, :func:`rdiv`
     - P
     - Y
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`dot`
     - N
     - Y
     - Y
     -
   * - :func:`drop`
     - Y
     - Y
     - Y
     -
   * - :func:`drop_duplicates`
     - P
     - Y
     - Y
     - SQL engine: maintaining original order of data not supported.
   * - :func:`droplevel`
     - N
     - Y
     - Y
     -
   * - :func:`dropna`
     - Y
     - Y
     - Y
     -
   * - :func:`duplicated`
     - N
     - Y
     - Y
     -
   * - :func:`empty`
     - Y
     - Y
     - Y
     -
   * - :func:`eq`, :func:`ne`
     - P
     - Y
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`explode`
     - Y
     - Y
     - Y
     -
   * - :func:`eval`
     - Y
     - Y
     - Y
     -
   * - :func:`fillna`
     - P
     - Y
     - Y
     - SQL engine: argument ``downcast``, ``limit`` and ``method`` not supported. ``axis==1`` not supported.
   * - :func:`filter`
     - N
     - Y
     - Y
     -
   * - :func:`first_valid_index`
     - N
     - Y
     - Y
     -
   * - :func:`floordiv`, :func:`rfloordiv`
     - P
     - Y
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`from_dict`
     - N
     - Y
     - Y
     -
   * - :func:`from_records`
     - N
     - Y
     - Y
     -
   * - :func:`ge`, :func:`gt`
     - P
     - Y
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`get`
     - Y
     - Y
     - Y
     -
   * - :func:`head`
     - Y
     - Y
     - Y
     -
   * - :func:`hist`
     - Y
     - Y
     - Y
     -
   * - ``df[item]`` (or ```__getitem__``)
     - P
     - Y
     - Y
     - SQL engine: when index of ``item`` and ``df`` are different, boolean indexing not supported.
   * - :func:`iat`
     - N
     - Y
     - Y
     -
   * - :func:`idxmax`
     - N
     - Y
     - Y
     -
   * - :func:`idxmin`
     - N
     - Y
     - Y
     -
   * - :func:`iloc`
     - P
     - Y
     - Y
     - SQL engine: single row selection with ``df.iloc[1]`` not supported. Non-continuous indexes or negative indexes (for instance, ``df.iloc[[1, 3]]``, ``df.iloc[1:10:2]`` or ``df.iloc[-3:]``) not supported.
   * - :func:`insert`
     - N
     - Y
     - Y
     -
   * - ``__invert__``
     - Y
     - Y
     - Y
     -
   * - :func:`isin`
     - P
     - Y
     - Y
     - SQL engine: index input not supported.
   * - :func:`isna`, :func:`notna`
     - Y
     - Y
     - Y
     -
   * - :func:`isnull`, :func:`notnull`
     - Y
     - Y
     - Y
     -
   * - :func:`items`
     - Y
     - Y
     - Y
     -
   * - :func:`iterrows`
     - Y
     - Y
     - Y
     -
   * - :func:`itertuples`
     - Y
     - Y
     - Y
     -
   * - :func:`join`
     - P
     - Y
     - Y
     -
   * - :func:`kurtosis`
     - N
     - Y
     - Y
     -
   * - :func:`last_valid_index`
     - N
     - Y
     - Y
     -
   * - :func:`le`, :func:`lt`
     - P
     - Y
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`loc`
     - N
     - Y
     - Y
     -
   * - :func:`map`
     - P
     - Y
     - Y
     - SQL engine: argument ``arg`` only supports functions and non-derivative dicts with simple scalars.
   * - :func:`mask`
     - N
     - Y
     - Y
     -
   * - :func:`max`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`mean`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`median`
     - Y
     - N
     - Y
     -
   * - :func:`melt`
     - Y
     - Y
     - Y
     -
   * - :func:`memory_usage`
     - N
     - Y
     - Y
     -
   * - :func:`merge`
     - P
     - Y
     - Y
     - SQL engine: argument ``indicator`` and ``validate`` not supported.
   * - :func:`min`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`mod`, :func:`rmod`
     - P
     - Y
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`mul`, :func:`rmul`
     - P
     - Y
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`nunique`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` or ``dropna==False`` not supported.
   * - :func:`pct_change`
     - N
     - Y
     - Y
     -
   * - :func:`pivot`
     - Y
     - Y
     - Y
     -
   * - :func:`pivot_table`
     - P
     - P
     - Y
     - SQL and DPE engine: argument ``margins`` not supported.
   * - :func:`plot`
     - Y
     - Y
     - Y
     -
   * - :func:`pop`
     - Y
     - Y
     - Y
     -
   * - :func:`pow`, :func:`rpow`
     - P
     - Y
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`prod`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`quantile`
     - Y
     - Y
     - Y
     -
   * - :func:`query`
     - Y
     - Y
     - Y
     -
   * - :func:`reindex`
     - P
     - Y
     - Y
     -
   * - :func:`reindex_like`
     - P
     - Y
     - Y
     -
   * - :func:`rename`
     - P
     - Y
     - Y
     - SQL engine: argument ``index`` not supported.
   * - :func:`rename_axis`
     - Y
     - Y
     - Y
     -
   * - :func:`reorder_levels`
     - N
     - Y
     - Y
     -
   * - :func:`replace`
     - P
     - Y
     - Y
     - SQL engine: when the argument ``regex`` is True, ``list`` or ``dict`` typed ``to_replace`` not supported. ``list`` or ``dict`` typed ``regex`` argument not supported.
   * - :func:`sample`
     - P
     - Y
     - Y
     - SQL engine: argument ``replace`` and ``weights`` not supported. ``axis==1`` or ``frac>1`` not supported.
   * - :func:`select_dtypes`
     - Y
     - Y
     - Y
     -
   * - :func:`sem`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`set_axis`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==0`` not supported.
   * - :func:`set_index`
     - Y
     - Y
     - Y
     -
   * - ``df[item] = value`` (or ```__setitem__``)
     - P
     - Y
     - Y
     - SQL engine: not supported when ``item`` is callable or DataFrame / Series or index of ``df`` and ``item`` are different.
   * - :func:`shift`
     - N
     - Y
     - Y
     -
   * - :func:`skew`
     - N
     - Y
     - Y
     -
   * - :func:`sort_index`
     - P
     - Y
     - Y
     - SQL engine: ``axis==1`` or ``na_position=='last'`` not supported.
   * - :func:`sort_values`
     - P
     - Y
     - Y
     - SQL engine: ``axis==1`` or ``na_position=='last'`` not supported.
   * - :func:`stack`
     - N
     - Y
     - Y
     -
   * - :func:`std`
     - P
     - Y
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`sub`, :func:`rsub`
     - P
     - Y
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`sum`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`swaplevel`
     - N
     - Y
     - Y
     -
   * - :func:`tail`
     - N
     - Y
     - Y
     -
   * - :func:`take`
     - N
     - Y
     - Y
     -
   * - :func:`transform`
     - P
     - Y
     - Y
     - SQL engine: not supported when ``axis==1`` and ``func`` is dict-like or list-like.
   * - :func:`transpose`
     - N
     - Y
     - Y
     -
   * - :func:`truediv`, :func:`rtruediv`
     - P
     - Y
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`unstack`
     - N
     - Y
     - Y
     -
   * - :func:`value_counts`
     - N
     - Y
     - Y
     -
   * - :func:`var`
     - P
     - Y
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`where`
     - N
     - Y
     - Y
     -
   * - :func:`xs`
     - N
     - Y
     - Y
     -


Index API
~~~~~~~~~~
.. currentmodule:: maxframe.dataframe.Index

.. list-table::
   :header-rows: 1

   * - API
     - SQL Engine
     - DPE
     - SPE
     - Details
   * - :func:`all`
     - N
     - Y
     - Y
     -
   * - :func:`any`
     - N
     - Y
     - Y
     -
   * - :func:`append`
     - Y
     - Y
     - Y
     -
   * - :func:`astype`
     - P
     - Y
     - Y
     - SQL engine: converting to categorical types not supported.
   * - :func:`count`
     - N
     - Y
     - Y
     -
   * - :func:`drop`
     - Y
     - Y
     - Y
     -
   * - :func:`drop_duplicates`
     - P
     - Y
     - Y
     - SQL engine: maintaining original order of data not supported.
   * - :func:`droplevel`
     - N
     - Y
     - Y
     -
   * - :func:`dropna`
     - Y
     - Y
     - Y
     -
   * - :func:`duplicated`
     - N
     - Y
     - Y
     -
   * - :func:`empty`
     - Y
     - Y
     - Y
     -
   * - :func:`fillna`
     - P
     - Y
     - Y
     - SQL engine: argument ``downcast``, ``limit`` and ``method`` not supported.
   * - ``series[item]`` (or ```__getitem__``)
     - N
     - Y
     - Y
     -
   * - :func:`get_level_values`
     - N
     - Y
     - Y
     -
   * - :func:`iloc`
     - P
     - Y
     - Y
     - SQL engine: Non-continuous indexes or negative indexes (for instance, ``df.iloc[[1, 3]]``, ``df.iloc[1:10:2]`` or ``df.iloc[-3:]``) not supported.
   * - :func:`isin`
     - N
     - Y
     - Y
     -
   * - :func:`isna`, :func:`notna`
     - Y
     - Y
     - Y
     -
   * - :func:`isnull`, :func:`notnull`
     - Y
     - Y
     - Y
     -
   * - :func:`map`
     - P
     - Y
     - Y
     - SQL engine: argument ``arg`` only supports functions and non-derivative dicts with simple scalars.
   * - :func:`max`
     - N
     - Y
     - Y
     -
   * - :func:`memory_usage`
     - N
     - Y
     - Y
     -
   * - :func:`mean`
     - N
     - Y
     - Y
     -
   * - :func:`min`
     - N
     - Y
     - Y
     -
   * - :func:`nunique`
     - N
     - Y
     - Y
     -
   * - :func:`reindex`
     - N
     - Y
     - Y
     -
   * - :func:`rename`
     - Y
     - Y
     - Y
     -
   * - :func:`set_names`
     - Y
     - Y
     - Y
     -
   * - :func:`shift`
     - N
     - Y
     - Y
     -
   * - :func:`sort_values`
     - N
     - Y
     - Y
     -
   * - :func:`to_frame`
     - Y
     - Y
     - Y
     -
   * - :func:`to_series`
     - Y
     - Y
     - Y
     -
   * - :func:`unique`
     - N
     - Y
     - Y
     -
   * - :func:`value_counts`
     - N
     - Y
     - Y
     -
   * - :func:`where`
     - N
     - Y
     - Y
     -
