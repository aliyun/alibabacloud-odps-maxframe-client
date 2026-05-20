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

import re
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

from odps import ODPS
from odps.models import Resource as ODPSResource

from maxframe import opcodes
from maxframe.core import ENTITY_TYPE, EntityData, OutputType
from maxframe.core.operator import ObjectOperator
from maxframe.learn.core import LearnOperatorMixin
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    FieldTypes,
    Int32Field,
    ListField,
    StringField,
)
from maxframe.utils import find_objects, replace_objects

# Pre-compiled regex for slash-format model name: projects/<project>/schemas/<schema>/models/<model>
# The model name part (.+) allows dots and other characters in model names (e.g. "qwen3.6-plus")
_MODEL_NAME_RESOURCE_REGEX = re.compile(
    r"^projects/([^/]+)/schemas/([^/]+)/models/(.+)$"
)

_odps_model_classes: Set["ODPSModelMixin"] = set()


def register_odps_model(model_cls: "ODPSModelMixin"):
    _odps_model_classes.add(model_cls)
    return model_cls


class ReadODPSModel(ObjectOperator, LearnOperatorMixin):
    _op_type_ = opcodes.READ_ODPS_MODEL
    # We need to change location, thus the serialized fields cannot be cached
    _cache_primitive_serial = False

    model_name = StringField("model_name", default=None)
    model_version = StringField("model_version", default=None)
    format = StringField("format", default=None)
    location = StringField("location", default=None)
    storage_options = DictField("storage_options", default=None)
    source_type = StringField("source_type", default=None)
    tasks = ListField("tasks", FieldTypes.string, default=None)
    options = DictField(
        "options",
        key_type=FieldTypes.string,
        value_type=FieldTypes.string,
        default=None,
    )
    inference_parameters = DictField(
        "inference_parameters",
        key_type=FieldTypes.string,
        value_type=FieldTypes.string,
        default=None,
    )

    def has_custom_code(self) -> bool:
        return True

    def __call__(self):
        for model_cls in _odps_model_classes:
            ret = model_cls._build_odps_source_model(self)
            if ret is not None:
                return ret
        raise ValueError(
            f"Model '{self.model_name}' with format '{self.format}' is not supported. "
            f"Supported formats include: BOOSTED_TREE_*, LLM, MLLM"
        )


class ReadODPSResource(ObjectOperator, LearnOperatorMixin):
    _op_type_ = opcodes.READ_ODPS_RESOURCE

    resource_path = StringField("resource_path", default=None)
    load_method = StringField("load_method", default=None)

    def has_custom_code(self) -> bool:
        return True

    def __call__(self):
        self._output_types = [OutputType.object]
        return self.new_tileable([], shape=())


class ToODPSModel(ObjectOperator, LearnOperatorMixin):
    _op_type_ = opcodes.TO_ODPS_MODEL
    # We need to change location, thus the serialized fields cannot be cached
    _cache_primitive_serial = False

    model_name = StringField("model_name", default=None)
    model_version = StringField("model_version", default=None)
    training_info = AnyField("training_info", default=None)
    params = AnyField("params", default=None)
    format = StringField("format", default=None)
    lifecycle = Int32Field("lifecycle", default=None)
    version_lifecycle = Int32Field("version_lifecycle", default=None)
    description = StringField("description", default=None)
    version_description = StringField("version_description", default=None)
    create_model = BoolField("create_model", default=True)
    set_default_version = BoolField("set_default_version", default=True)
    location = StringField("location", default=None)
    storage_options = DictField("storage_options", default=None)

    def __init__(self, **kw):
        super().__init__(_output_types=[OutputType.object], **kw)

    @classmethod
    def _set_inputs(cls, op: "ToODPSModel", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)

        if isinstance(op.training_info, ENTITY_TYPE):
            has_training_info = True
            op.training_info = inputs[0]
        else:
            has_training_info = False

        tileables = find_objects([op.params], ENTITY_TYPE)
        param_pos = int(has_training_info)
        replaces = dict(zip(tileables, inputs[param_pos:]))
        [op.params] = replace_objects([op.params], replaces)

    def can_fuse_with_custom_code(self) -> bool:
        return False

    def __call__(self, training_info, params):
        inputs = []
        if isinstance(training_info, ENTITY_TYPE):
            inputs.append(training_info)

        self.training_info = training_info
        self.params = params
        inputs.extend(find_objects([params], ENTITY_TYPE))
        return self.new_tileable(inputs, shape=())


class ODPSModelMixin:
    class ODPSModelInfo(NamedTuple):
        model_format: str
        model_params: Any

    @classmethod
    def _build_odps_source_model(cls, op: ReadODPSModel) -> Any:
        return None

    def _get_odps_model_info(self) -> ODPSModelInfo:
        raise NotImplementedError

    def to_odps_model(
        self,
        model_name: str,
        model_version: str = None,
        schema: str = None,
        project: str = None,
        lifecycle: Optional[int] = None,
        version_lifecycle: Optional[int] = None,
        description: Optional[str] = None,
        version_description: Optional[str] = None,
        create_model: bool = True,
        set_default_version: bool = False,
        location: Optional[str] = None,
        storage_options: Dict[str, Any] = None,
    ):
        """
        Save trained model to MaxCompute.

        Parameters
        ----------
        model_name : str, optional
            Name of the model. Can be a fully qualified name with format
            "project.schema.model" or just "model" if project and schema are
            specified separately.
        model_version : str, optional
            Version of the model. If not provided, a default version will be used.
        schema : str, optional
            Schema name where the model will be stored. If not provided and
            project is specified, "default" schema will be used.
        project : str, optional
            Project name where the model will be stored.
        lifecycle : int, optional
            Lifecycle of the model in days. After this period, the model will
            be automatically deleted.
        version_lifecycle : int, optional
            Lifecycle of the model version in days. After this period, the
            model version will be automatically deleted.
        description : str, optional
            Description of the model.
        version_description : str, optional
            Description of the model version.
        create_model : bool, default True
            Whether to create the model if it doesn't exist.
        set_default_version : bool, default False
            Whether to set this version as the default version of the model.
        location : str, optional
            Storage location for the model. If specified, the model can be stored
            into a customized location. Can be an OSS path with format
            oss://endpoint/bucket/path.
        storage_options : dict, optional
            Extra options for storage, such as role_arn or policy for OSS storage.

        Returns
        -------
        Scalar
            A scalar that can be executed to save the model.

        Examples
        --------
        First we fit an XGBoost model.

        >>> import maxframe.dataframe as md
        >>> from maxframe.learn.datasets import make_classification
        >>> from maxframe.learn.contrib.xgboost import XGBClassifier
        >>> X, y = make_classification(1000, n_features=10, n_classes=2)
        >>> cols = [f"f{idx}" for idx in range(10)]
        >>> clf = XGBClassifier(n_estimators=10)
        >>> X_df = md.DataFrame(X, columns=cols)
        >>> clf.fit(X_df, y)

        Trigger execution and save model with fully qualified name.

        >>> clf.to_odps_model(model_name="project.schema.my_model",
        ...                   model_version="v1.0").execute()

        You can also save model with a customized path. Need to change `<my_bucket>`
        and `<user_id>` into your own bucket and user ID.

        >>> clf.to_odps_model(model_name="project.schema.my_model",
        ...                   model_version="v1.0",
        ...                   location="oss://oss-cn-shanghai.aliyuncs.com/<my_bucket>/model_name",
        ...                   storage_options={
        ...                       "role_arn": "acs:ram::<user_id>:role/aliyunodpsdefaultrole"
        ...                   }).execute()
        """
        # Resolve project: fall back to the default ODPS project when not specified,
        # so that existing code calling to_odps_model(model_name="my_model") without
        # an explicit project continues to work.
        resolved_project = project or getattr(
            ODPS.from_global() or ODPS.from_environments(), "project", None
        )
        model_name = _build_odps_model_name(model_name, schema, resolved_project)
        model_info = self._get_odps_model_info()

        op = ToODPSModel(
            model_name=model_name,
            model_version=model_version,
            format=model_info.model_format,
            lifecycle=lifecycle,
            version_lifecycle=version_lifecycle,
            description=description,
            version_description=version_description,
            create_model=create_model,
            set_default_version=set_default_version,
            location=location,
            storage_options=storage_options,
        )
        return op(getattr(self, "training_info_"), model_info.model_params)


def parse_odps_model_name(model_name: str) -> Tuple[str, str, str]:
    """
    Parse a model name into (project, schema, model_name) components.

    Supports two formats:
    1. Slash format: ``projects/<project>/schemas/<schema>/models/<model>``
       The literal segments "projects", "schemas" and "models" are required.
    2. Dot format: ``project.schema.model_name`` or ``project.model_name``
       When there are 3+ dot-separated parts, the model name itself may
       contain dots (e.g. "qwen3.6-plus"), so parts[2:] are joined back.

    Parameters
    ----------
    model_name : str
        The full model name to parse.

    Returns
    -------
    Tuple[str, str, str]
        A tuple of (project, schema, model_short_name).
        For slash format, values are extracted directly.
        For dot format with 2 parts, schema defaults to "default".
        For dot format with 1 part, project is "" and schema is "default".

    Raises
    ------
    ValueError
        If slash format is used but doesn't match the required structure.
    """
    if "/" in model_name:
        m = _MODEL_NAME_RESOURCE_REGEX.match(model_name)
        if not m:
            raise ValueError(
                f"Model name format mismatch: {model_name!r}. "
                "Expected 'projects/<project>/schemas/<schema>/models/<model>'"
            )
        return m.group(1), m.group(2), m.group(3)

    # Parse full name in old dot format
    parts = [part.strip() for part in model_name.split(".") if part.strip()]
    if len(parts) >= 3:
        return parts[0], parts[1], ".".join(parts[2:])
    elif len(parts) == 2:
        return parts[0], "default", parts[1]
    else:
        return "", "default", parts[0] if parts else ""


def _build_odps_model_name(model_name: str, schema: str, project: str = None):
    """
    Build a fully-qualified model name in slash format.

    Always produces ``projects/<project>/schemas/<schema>/models/<model>`` format.
    If the input is already a valid slash-format name, it is returned as-is.
    Otherwise, the input is treated as a short model name and the
    ``project`` / ``schema`` arguments are used to build the full name
    (``schema`` defaults to ``"default"``).

    ``project`` is required when the input is a short (non-slash) name,
    because the round-trip through ``parse_odps_model_name`` must be
    lossless: a short name that contains dots (e.g. "qwen3.6-plus") cannot
    be correctly parsed back without the slash-format wrapper.

    Parameters
    ----------
    model_name : str
        Short model name or an already-qualified slash-format name.
    schema : str
        Schema name (defaults to ``"default"`` inside the built name).
    project : str, optional
        Project name. Required when *model_name* is a short name.
        Ignored when *model_name* is already in slash format.

    Raises
    ------
    ValueError
        If *model_name* is a short name and *project* is not provided, or
        if a slash-format name does not match the expected structure.
    """
    if "/" in model_name:
        if not _MODEL_NAME_RESOURCE_REGEX.match(model_name):
            raise ValueError(
                f"Model name format mismatch: {model_name!r}. "
                "Expected 'projects/<project>/schemas/<schema>/models/<model>'"
            )
        return model_name

    # model_name is a short name (may contain dots like "qwen3.6-plus")
    if not project:
        raise ValueError(
            "project is required when building a fully-qualified model name "
            f"from a short name {model_name!r}"
        )

    return f"projects/{project}/schemas/{schema or 'default'}/models/{model_name}"


def read_odps_model(
    model_name: str,
    schema: str = None,
    project: str = None,
    model_version: str = None,
    odps_entry: ODPS = None,
):
    odps_entry = odps_entry or ODPS.from_global() or ODPS.from_environments()
    if not hasattr(odps_entry, "get_model"):
        raise RuntimeError("Need to install pyodps>=0.11.5 to use read_odps_model")

    model_obj = odps_entry.get_model(model_name, project, schema)
    if model_version:
        model_obj = model_obj.versions[model_version]
    # check if model exists
    model_obj.reload()

    full_model_name = _build_odps_model_name(
        model_name, schema, project or odps_entry.project
    )
    location = model_obj.path
    format_ = model_obj.type.value
    source_type = model_obj.source_type.value

    op = ReadODPSModel(
        model_name=full_model_name,
        model_version=model_version,
        location=location,
        format=format_,
        source_type=source_type,
        options=getattr(model_obj, "options", None) or {},
        tasks=getattr(model_obj, "tasks", None) or [],
        inference_parameters=getattr(model_obj, "inference_parameters", None) or {},
    )
    return op()


def read_odps_resource(
    resource: Union[str, ODPSResource],
    load_method: Optional[str] = "pickle",
    odps_entry: ODPS = None,
):
    odps_entry = odps_entry or ODPS.from_global() or ODPS.from_environments()
    if not isinstance(resource, ODPSResource):
        resource = odps_entry.get_resource(resource)

    schema_name = resource.schema.name if resource.schema else None
    if odps_entry.is_schema_namespace_enabled():
        schema_name = schema_name or odps_entry.schema or "default"
    full_res_name = ODPSResource.build_full_resource_name(
        resource.name, resource.project.name, schema_name
    )

    if load_method and load_method not in ("pickle", "joblib"):
        raise ValueError(
            f"load_method must be one of 'pickle' or 'joblib', got {load_method}"
        )

    op = ReadODPSResource(
        resource_path=full_res_name,
        load_method=load_method,
    )
    return op()
