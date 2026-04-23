# Copyright 1999-2026 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Import commonly used utilities for backward compatibility
# Only import what's actually used by other modules to avoid circular imports

# Import from _utils_c (Cython modules)
from ._utils_c import (  # Common utilities from _utils_c
    NamedType,
    Timer,
    TypeDispatcher,
    ceildiv,
    get_user_call_point,
    new_random_id,
    register_tokenizer,
    reset_id_random_seed,
    to_binary,
    to_str,
    to_text,
    tokenize,
    tokenize_int,
)

# Import asyncio utilities
from .aio import (
    ToThreadCancelledError,
    ToThreadMixin,
    call_with_retry,
    create_sync_primitive,
    relay_future,
    wait_http_response,
)

# Import collection utilities
from .collections_ import (
    AttributeDict,
    LRUDict,
    find_objects,
    flatten,
    is_empty,
    replace_objects,
    stack_back,
)

# Data type functions (most commonly imported)
from .datatypes import (
    arrow_type_from_str,
    is_arrow_dtype_supported,
    is_bool_dtype,
    is_datetime64_dtype,
    is_string_dtype,
    make_dtype,
    make_dtypes,
    wrap_arrow_dtype,
)

# Decorators and functional utilities
from .functional import (
    deprecate_positional_args,
    enter_current_session,
    get_func_token,
    ignore_warning,
    implements,
    quiet_stdio,
    skip_na_call,
    unwrap_function,
)

# Import ODPS utilities
from .odps import (
    add_survey_log,
    build_session_volume_name,
    build_temp_intermediate_table_name,
    build_temp_table_name,
    config_odps_default_options,
    get_default_table_properties,
    get_odps_dlf_table,
    submit_survey_logs,
    sync_pyodps_options,
    update_wlm_quota_settings,
)

# Import serialization utilities
from .serialization import (
    deserialize_serializable,
    on_deserialize_shape,
    on_serialize_nsplits,
    on_serialize_numpy_type,
    on_serialize_shape,
    serialize_serializable,
)

# Re-export no_default for convenience
# Core utility classes and functions
from .utils import (
    KeyLogWrapper,
    ModulePlaceholder,
    NoDefault,
    PatchableMixin,
    ServiceLoggerAdapter,
    adapt_docstring,
    atomic_writer,
    cache_tileables,
    calc_nsplits,
    check_unexpected_kwargs,
    classproperty,
    collect_leaf_operators,
    combine_error_message_and_traceback,
    copy_if_possible,
    copy_tileables,
    dataslots,
    estimate_pandas_size,
    estimate_table_size,
    extract_class_name,
    format_timeout_params,
    generate_unique_id,
    get_handler_timeout_value,
    get_item_if_scalar,
    get_pd_option,
    get_python_tag,
    has_unknown_shape,
    is_full_slice,
    lazy_import,
    new_random_id,
    no_default,
    np_release_version,
    parse_readable_size,
    parse_size_to_megabytes,
    pd_option_context,
    pd_release_version,
    prevent_called_from_pandas,
    random_ports,
    remove_suffix,
    sbytes,
    str_to_bool,
    stringify_path,
    to_hashable,
    trait_from_env,
    url_path_join,
    validate_and_adjust_resource_ratio,
)
