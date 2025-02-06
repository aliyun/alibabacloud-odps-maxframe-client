# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

# retry consts
EMPTY_RESPONSE_RETRY_COUNT = 5

# Restful Service
RESTFUL_SESSION_INSECURE_SCHEME = "mf"
RESTFUL_SESSION_SECURE_SCHEME = "mfs"

# ODPS
ODPS_SESSION_INSECURE_SCHEME = "mfo"
ODPS_SESSION_SECURE_SCHEME = "mfos"

# ODPS MaxFrameTask consts
MAXFRAME_DEFAULT_PROTOCOL = "v1"
# method names
MAXFRAME_TASK_CREATE_SESSION_METHOD = "CREATE_SESSION"
MAXFRAME_TASK_GET_SESSION_METHOD = "GET_SESSION"
MAXFRAME_TASK_DELETE_SESSION_METHOD = "DELETE_SESSION"
MAXFRAME_TASK_SUBMIT_DAG_METHOD = "SUBMIT_DAG"
MAXFRAME_TASK_GET_DAG_INFO_METHOD = "GET_DAG_INFO"
MAXFRAME_TASK_CANCEL_DAG_METHOD = "CANCEL_DAG"
MAXFRAME_TASK_DECREF_METHOD = "DECREF"
# serialization methods
MAXFRAME_OUTPUT_MAXFRAME_FORMAT = "maxframe_v1"
MAXFRAME_OUTPUT_JSON_FORMAT = "json"
MAXFRAME_OUTPUT_MSGPACK_FORMAT = "msgpack"
