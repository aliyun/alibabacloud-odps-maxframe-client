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

from typing import Any, Dict, List, NamedTuple, Optional, Set

from odps import ODPS

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData, OutputType
from ...core.operator import ObjectOperator
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    Int32Field,
    StringField,
)
from ...utils import find_objects, replace_objects
from ..core import LearnOperatorMixin

_odps_model_classes: Set["ODPSModelMixin"] = set()


def register_odps_model(model_cls: "ODPSModelMixin"):
    _odps_model_classes.add(model_cls)
    return model_cls


class ReadODPSModel(ObjectOperator, LearnOperatorMixin):
    _op_type_ = opcodes.READ_ODPS_MODEL

    model_name = StringField("model_name", default=None)
    model_version = StringField("model_version", default=None)
    format = StringField("format", default=None)
    location = StringField("location", default=None)
    storage_options = DictField("storage_options", default=None)

    def has_custom_code(self) -> bool:
        return True

    def __call__(self):
        if not self.format.startswith("BOOSTED_TREE_"):
            # todo support more model formats
            raise ValueError("Only support boosted tree format")
        for model_cls in _odps_model_classes:
            ret = model_cls._build_odps_source_model(self)
            if ret is not None:
                return ret
        raise ValueError(f"Model {self.model_name} not supported")


class ToODPSModel(ObjectOperator, LearnOperatorMixin):
    _op_type_ = opcodes.TO_ODPS_MODEL

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
        model_name = _build_odps_model_name(model_name, schema, project)
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


def _build_odps_model_name(model_name: str, schema: str, project: str = None):
    if "." not in model_name:
        if project and not schema:
            schema = "default"
        if schema:
            model_name = f"{schema}.{model_name}"
        if project:
            model_name = f"{project}.{model_name}"
    return model_name


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

    full_model_name = _build_odps_model_name(model_name, schema, project)
    location = model_obj.path
    format = model_obj.type.value
    op = ReadODPSModel(
        model_name=full_model_name,
        model_version=model_version,
        location=location,
        format=format,
    )
    return op()
