.. _examples_index:

.. rst-class:: mf-examples-daft-page

Examples
========

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: mf-examples-daft-grid

   .. grid-item-card:: Multimodal image feature pipeline (OSS + embedding)
      :link: qwen3_vl_oss_image_embedding
      :link-type: doc
      :text-align: left
      :class-card: mf-examples-daft-card

      .. raw:: html

         <span class="mf-example-available">Available at MaxFrame 2.6.0</span>

      Build reusable image feature assets with distributed preprocessing, DashScope multimodal embedding, and vector persistence in MaxCompute.

   .. grid-item-card:: Video pipeline: frame extraction -> labeling -> embedding
      :link: maxframe_video_pipeline_best_practice
      :link-type: doc
      :text-align: left
      :class-card: mf-examples-daft-card

      .. raw:: html

         <span class="mf-example-available">Available at MaxFrame 2.6.0</span>

      End-to-end video processing on DPE using a custom FFmpeg image: frame extraction, multimodal labeling, and embedding generation.

   .. grid-item-card:: Bailian LLM calling tutorial
      :link: maxframe_ai_function_bailian_demo
      :link-type: doc
      :text-align: left
      :class-card: mf-examples-daft-card

      .. raw:: html

         <span class="mf-example-available">Available at MaxFrame 2.6.0</span>

      Access pre-registered models in ``BIGDATA_PUBLIC_MODELSET`` with ``read_odps_model()``, then run ``embed()`` and ``generate()`` at scale on DPE.

   .. grid-item-card:: MaxFrame multimodal audio operators practice
      :link: multimodal_audio_maxframe
      :link-type: doc
      :text-align: left
      :class-card: mf-examples-daft-card

      .. raw:: html

         <span class="mf-example-available">Available at MaxFrame 2.7.0</span>

      Process OSS audio directly through ``.url.download()`` and ``.audio.*`` operators for decode, language detection, transcription, and VAD.

   .. grid-item-card:: OpenLake multimodal images (Object Table + Paimon)
      :link: openlake_multimodal_maxframe
      :link-type: doc
      :text-align: left
      :class-card: mf-examples-daft-card

      Catalog OSS images with an object table, resize in a distributed MaxFrame UDF, and write to Paimon on DLF.

   .. grid-item-card:: PDF text parsing and Bailian embedding
      :link: pdf_text_embedding
      :link-type: doc
      :text-align: left
      :class-card: mf-examples-daft-card

      .. raw:: html

         <span class="mf-example-available">Available at MaxFrame 2.6.0</span>

      Extract text from OSS PDFs with ``apply_chunk``, split it into semantic chunks, generate Bailian embeddings, and write vector features to MaxCompute.

..
   If you add or reorder entries below, run ``make clean html`` (or ``make html-fresh``)
   once so already-built example pages regenerate their primary sidebar; incremental
   ``make html`` can leave stale ``bd-sidenav`` HTML for older tutorial files.

.. toctree::
   :hidden:
   :maxdepth: 1

   qwen3_vl_oss_image_embedding
   multimodal_audio_maxframe
   openlake_multimodal_maxframe
   maxframe_video_pipeline_best_practice
   maxframe_ai_function_bailian_demo
   pdf_text_embedding
