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

import functools
import shlex
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
from odps.models import Function as ODPSFunctionObj
from odps.models import Resource as ODPSResourceObj

from maxframe.config import options
from maxframe.config.validators import is_positive_integer
from maxframe.core.mode import is_mock_mode
from maxframe.serialization import load_member
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    FieldTypes,
    ListField,
    Serializable,
    StringField,
)
from maxframe.typing_ import PandasDType
from maxframe.utils import (
    extract_class_name,
    make_dtype,
    parse_size_to_megabytes,
    tokenize,
    unwrap_function,
    validate_and_adjust_resource_ratio,
)


class PythonPackOptions(Serializable):
    _key_args = ("force_rebuild", "prefer_binary", "pre_release", "no_audit_wheel")

    key = StringField("key")
    requirements = ListField("requirements", FieldTypes.string, default_factory=list)
    force_rebuild = BoolField("force_rebuild", default=False)
    prefer_binary = BoolField("prefer_binary", default=False)
    pre_release = BoolField("pre_release", default=False)
    pack_instance_id = StringField("pack_instance_id", default=None)
    no_audit_wheel = BoolField("no_audit_wheel", default=False)

    def __init__(self, key: str = None, **kw):
        super().__init__(key=key, **kw)
        if self.key is None:
            args = {k: getattr(self, k) for k in self._key_args}
            self.key = tokenize(set(self.requirements), args)

    def __repr__(self):
        args_str = " ".join(f"{k}={getattr(self, k)}" for k in self._key_args)
        return f"<PythonPackOptions {self.requirements} {args_str}>"


class FsMountOptions(Serializable):
    path = StringField("path")
    mount_path = StringField("mount_path")
    storage_options = DictField(
        "storage_options", FieldTypes.string, FieldTypes.any, default_factory=dict
    )

    def __repr__(self):
        return f"<FsMountOptions {self.path} -> {self.mount_path}>"

    def validate(self) -> None:
        if not self.path or not isinstance(self.path, str):
            raise ValueError("A valid path string is required.")
        if not self.mount_path or not isinstance(self.mount_path, str):
            raise ValueError("A valid mount_path string is required.")

        # Check authentication: either role_arn or ak/sk must be provided
        storage_opts = self.storage_options or {}
        has_role_arn = bool(storage_opts.get("role_arn"))
        has_ak_sk = bool(
            storage_opts.get("access_key_id") and storage_opts.get("access_key_secret")
        )
        if not has_role_arn and not has_ak_sk:
            raise ValueError(
                "Authentication credentials required in storage_options: "
                "either 'role_arn' or 'access_key_id'/'access_key_secret' must be provided."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "mount_path": self.mount_path,
            "storage_options": self.storage_options,
        }

    @classmethod
    def from_legacy_dict(cls, config: Dict[str, Any]) -> "FsMountOptions":
        # Reconstruct path: oss://{endpoint}/{bucket}/{prefix}
        endpoint = config.get("oss_endpoint", "")
        bucket = config.get("oss_bucket", "")
        prefix = config.get("oss_bucket_prefix", "")
        if endpoint and bucket:
            path = f"oss://{endpoint}/{bucket}"
            if prefix:
                path = f"{path}/{prefix}"
        else:
            path = ""

        # Collect storage_options from legacy fields
        storage_options = {}
        if config.get("role_arn"):
            storage_options["role_arn"] = config["role_arn"]
        if config.get("access_key_id"):
            storage_options["access_key_id"] = config["access_key_id"]
        if config.get("access_key_secret"):
            storage_options["access_key_secret"] = config["access_key_secret"]

        return cls(
            path=path,
            mount_path=config.get("mount_path", ""),
            storage_options=storage_options,
        )


class BuiltinFunction(Serializable):
    """
    Record reference for builtin functions. The function body
    will NOT be serialized when submitting jobs.
    """

    __slots__ = ("_func",)

    func_name = StringField("func_name", default=None)

    # func_name -> callable, for functions that cannot be imported by name.
    _registered: Dict[str, Callable] = {}

    @classmethod
    def register_function_body(cls, func_name: str, func: Callable) -> None:
        """Register *func* under *func_name*.

        A registered callable is resolved directly, without importing the
        name, and takes precedence over the import mechanism.  Registering
        is optional: an unregistered reference keeps resolving by import.
        """
        cls._registered[func_name] = func

    def __init__(self, func: Optional[Callable] = None, **kw):
        self._func = func
        if func is not None:
            # Unwrap ``MarkedFunction`` for name extraction: its
            # ``__module__`` resolves to the class's module
            # (``maxframe.udf``), not the raw function's, so the registered
            # name would point at the wrong module and worker reload would
            # recurse infinitely.  The ``MarkedFunction`` is retained in
            # ``_func`` for client-side scheduling-hint extraction.
            name_source = func.func if isinstance(func, MarkedFunction) else func
            func_name = extract_class_name(name_source)
            if "<" in func_name:
                raise ValueError("Cannot be a local or lambda function")
            kw["func_name"] = kw.get("func_name") or func_name
        super().__init__(**kw)

    @property
    def func(self):
        if getattr(self, "_func", None) is None:
            assert isinstance(self.func_name, str)
            registered = type(self)._registered.get(self.func_name)
            if registered is not None:
                self._func = registered
            else:
                self._func = load_member(self.func_name, type(self)).func
        return self._func

    @property
    def module(self):
        # use func_name instead of func itself to avoid it
        # been imported (might cause infinite recursion!)
        assert isinstance(self.func_name, str)
        return self.func_name.split("#")[0]

    def __getattr__(self, item):
        if item.startswith("_") and not item.endswith("_"):
            return super().__getattribute__(item)
        if item.startswith("__") and item.endswith("__"):
            # Dunder probes (``__wrapped__``, ``__closure__``, ``__globals__``,
            # ``__code__``) come from callable inspection -- closure entity
            # checks, scheduling-hint discovery, ``unwrap_function`` -- which runs
            # on the client while the graph is built.  A body-absent logical
            # reference must answer them as absent instead of importing
            # ``func_name``: the body usually lives in a private backend module
            # that is not importable there, and the graph only needs the
            # explicitly given output metadata.
            body = getattr(self, "_func", None)
            if body is None:
                body = type(self)._registered.get(self.func_name)
            if body is None:
                raise AttributeError(item)
        attr = getattr(self.func, item)
        # Wrap callable methods as BuiltinMethod only for a
        # builtin-decorated *class*: BuiltinMethod.func reloads+instantiates
        # the class, so wrapping a plain function's callable attrs (e.g.
        # inspect.unwrap descending __wrapped__, singledispatch.register)
        # would call the function with no args and break.
        if (
            callable(attr)
            and not isinstance(attr, type)
            and isinstance(self.func, type)
        ):
            return BuiltinMethod(func_name=self.func_name, method_name=item)
        return attr

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)


class BuiltinMethod(BuiltinFunction):
    """Wrap an instance method of a ``@builtin_function``-decorated class.

    Carries the class ``func_name`` + ``method_name``.  On the worker, the
    class is reloaded by name, instantiated, and the bound method is invoked
    -- preserving ``self`` binding for instance methods (e.g. ``CustomAgg``
    ``pre``/``post``) sent through ``transform_chunk``.

    ``isinstance(bm, BuiltinFunction)`` is ``True`` so ``has_custom_code``
    checks return ``False`` (no isolation sandbox).
    """

    method_name = StringField("method_name", default=None)

    def __init__(self, func_name=None, method_name=None, **kw):
        kw["func_name"] = func_name
        kw["method_name"] = method_name
        super().__init__(func=None, **kw)

    @property
    def func(self):
        # Resolve the @builtin_function-decorated class through the same path
        # as BuiltinFunction.func so a class registered only via
        # register_function_body is found before falling back to import.
        # BuiltinFunction (not type(self)) is used deliberately: this
        # property overrides BuiltinFunction.func, so recursing would loop.
        cls = BuiltinFunction(func_name=self.func_name).func
        instance = cls()  # instantiate
        return getattr(instance, self.method_name)  # bound method

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)


def builtin_function(func: Callable) -> BuiltinFunction:
    return BuiltinFunction(func=func)


class MarkedFunction(Serializable):
    func = AnyField("func", default=None)
    file_resources = ListField("resources", FieldTypes.string, default_factory=list)
    pythonpacks = ListField("pythonpacks", FieldTypes.reference, default_factory=list)
    expect_engine = StringField("expect_engine", default=None)
    expect_resources = DictField(
        "expect_resources", FieldTypes.string, default_factory=dict
    )
    gpu = BoolField("gpu", default=False)
    fs_mount = ListField("fs_mount", FieldTypes.reference, default_factory=list)
    public_network_whitelist = ListField(
        "public_network_whitelist", FieldTypes.string, default_factory=list
    )
    internal_network_whitelist = ListField(
        "internal_network_whitelist", FieldTypes.string, default_factory=list
    )
    vpc_network_link = StringField("vpc_network_link", default=None)
    # image_options configuration: {"name": "image_name"}
    image_options = DictField("image_options", FieldTypes.string, default=None)

    def __init__(self, func: Optional[Callable] = None, **kw):
        super().__init__(func=func, **kw)

    def __getattr__(self, item):
        return getattr(self.func, item)

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)

    def __get__(self, instance, owner=None):
        """Bind ``self`` to *instance* for method-style access.

        ``@with_running_options`` / ``@with_resources`` decorate a method
        by storing a ``MarkedFunction`` on the class.  Without a descriptor,
        ``instance.method`` returns the *unbound* ``MarkedFunction``, so a
        decorated ``CustomAgg.pre``/``.agg``/``.post`` (or any decorated
        instance method) drops ``self`` and crashes on call.

        Returning ``functools.partial(self, instance)`` binds the first
        positional argument to the instance while keeping the
        ``MarkedFunction`` discoverable via ``partial.func`` —
        ``discover_marked_functions`` already traverses ``functools.partial``
        into its ``.func``, so scheduling-hint extraction still finds the
        running options on the bound method.  When accessed on the class
        (``instance is None``) return ``self`` unchanged so class-level
        introspection (``cls.pre is CustomAgg.pre``) keeps working.
        """
        if instance is None:
            return self
        return functools.partial(self, instance)

    def __repr__(self):
        return f"<MarkedFunction {self.func!r}>"

    @property
    def __wrapped__(self):
        return self.func


class ODPSFunction(Serializable):
    __slots__ = ("_caller_type",)

    full_function_name = StringField("full_function_name", default=None)
    expect_engine = StringField("expect_engine", default=None)
    expect_resources = DictField(
        "expect_resources", FieldTypes.string, default_factory=dict
    )
    result_dtype = AnyField("result_dtype", default=None)

    def __init__(
        self,
        func,
        expect_engine: str = None,
        expect_resources: dict = None,
        dtype: PandasDType = None,
        **kw,
    ):
        full_function_name = None
        if isinstance(func, str):
            full_function_name = func
        elif isinstance(func, ODPSFunctionObj):
            func_parts = [func.project.name]
            if func.schema:
                func_parts.append(func.schema.name)
            func_parts.append(func.name)
            full_function_name = ":".join(func_parts)
        if full_function_name:
            kw["full_function_name"] = full_function_name

        if dtype is not None:
            kw["result_dtype"] = make_dtype(dtype)
        super().__init__(
            expect_engine=expect_engine, expect_resources=expect_resources, **kw
        )

    @property
    def __name__(self):
        return self.full_function_name.rsplit(":", 1)[-1]

    def _detect_caller_type(self) -> Optional[str]:
        if hasattr(self, "_caller_type"):
            return self._caller_type

        frame = sys._getframe(1)
        is_set = False
        while frame.f_back:
            f_mod = frame.f_globals.get("__name__")
            if f_mod and f_mod.startswith("maxframe.dataframe."):
                if f_mod.endswith(".map"):
                    self._caller_type, is_set = "map", True
                elif f_mod.endswith(".aggregation") or ".reduction." in f_mod:
                    self._caller_type, is_set = "agg", True
                if is_set:
                    return self._caller_type
            frame = frame.f_back
        return None

    def __call__(self, obj, *args, **kwargs):
        caller_type = self._detect_caller_type()
        if caller_type == "agg":
            return self._call_aggregate(obj, *args, **kwargs)
        raise NotImplementedError("Need to be referenced inside apply or map functions")

    def _call_aggregate(self, obj, *args, **kwargs):
        from maxframe.dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
        from maxframe.dataframe.reduction.custom_reduction import (
            build_custom_reduction_result,
        )

        if isinstance(obj, (DATAFRAME_TYPE, SERIES_TYPE)):
            return build_custom_reduction_result(obj, self)
        if is_mock_mode():
            ret = obj.iloc[0]
            if self.result_dtype:
                if hasattr(ret, "astype"):
                    ret = ret.astype(self.result_dtype)
                else:  # pragma: no cover
                    ret = np.array(ret).astype(self.result_dtype).item()
            return ret
        raise NotImplementedError("Need to be referenced inside apply or map functions")

    def __repr__(self):
        return f"<ODPSStoredFunction {self.full_function_name}>"

    @classmethod
    def wrap(cls, func):
        if isinstance(func, ODPSFunctionObj):
            return ODPSFunction(func)
        return func


def with_resources(
    *resources: Union[str, ODPSResourceObj], use_wrapper_class: bool = True
):
    """
    Make MaxCompute resource files available to a UDF.

    Parameters
    ----------
    *resources : str or odps.models.Resource
        Names or Resource objects for files already uploaded to MaxCompute.
        Resource objects retain their project and schema qualification.
    use_wrapper_class : bool, default True
        Wrap the callable in a MarkedFunction. If False, append to the callable's
        existing ``file_resources`` attribute instead.

    Returns
    -------
    callable
        Decorator that adds resources and returns the decorated callable.
        Repeated decorators append resources.

    Raises
    ------
    AttributeError
        If ``use_wrapper_class=False`` and the callable has no
        ``file_resources`` attribute.

    Notes
    -----
    DPE and MCSQL provide the files in the function's working directory.
    ``with_resource_libraries`` is an alias of this function.

    Examples
    --------
    >>> from maxframe.udf import with_resources
    >>> @with_resources("labels.txt")
    ... def label_count(batch):
    ...     with open("labels.txt", encoding="utf-8") as stream:
    ...         count = sum(1 for line in stream)
    ...     return batch.assign(label_count=count)
    """

    def res_to_str(res: Union[str, ODPSResourceObj]) -> str:
        if isinstance(res, str):
            return res
        res_parts = [res.project.name]
        if res.schema:
            res_parts.extend(["schemas", res.schema])
        res_parts.extend(["resources", res.name])
        return "/".join(res_parts)

    def func_wrapper(func):
        str_resources = [res_to_str(r) for r in resources]
        if not use_wrapper_class:
            existing = getattr(func, "file_resources") or []
            func.file_resources = existing + str_resources
            return func

        if isinstance(func, MarkedFunction):
            func.file_resources = func.file_resources + str_resources
            return func
        return MarkedFunction(func, file_resources=str_resources)

    return func_wrapper


def with_python_requirements(
    *requirements: str,
    force_rebuild: bool = False,
    prefer_binary: bool = False,
    pre_release: bool = False,
    no_audit_wheel: bool = False,
):
    """
    Declare Python packages needed by a remote UDF.

    Parameters
    ----------
    *requirements : str
        Package requirements, optionally with version constraints, such as
        ``"Pillow==10.4.0"``. Packages must support the remote Python ABI.
    force_rebuild : bool, default False
        Rebuild the PythonPack environment instead of reusing a matching build.
    prefer_binary : bool, default False
        Prefer binary wheels over source distributions.
    pre_release : bool, default False
        Allow pre-release package versions.
    no_audit_wheel : bool, default False
        Skip the wheel compatibility audit. The wheel must still be compatible
        with the remote runtime.

    Returns
    -------
    callable
        Decorator that attaches PythonPack requirements to a callable.
        Repeated decorators append requirement groups.

    Raises
    ------
    ValueError
        If a requirement contains malformed shell-style quoting.

    Notes
    -----
    DPE and MCSQL build or reuse the package environment before running the UDF.
    Installing a package on the client does not install it remotely.

    Examples
    --------
    >>> from maxframe.udf import with_python_requirements
    >>> @with_python_requirements("Pillow==10.4.0", prefer_binary=True)
    ... def image_widths(batch):
    ...     from io import BytesIO
    ...     from PIL import Image
    ...     return batch.assign(
    ...         width=[Image.open(BytesIO(raw)).width for raw in batch["image_bytes"]]
    ...     )
    """
    result_req = []
    for req in requirements:
        result_req.extend(shlex.split(req))

    def func_wrapper(func):
        pack_item = PythonPackOptions(
            requirements=requirements,
            force_rebuild=force_rebuild,
            prefer_binary=prefer_binary,
            pre_release=pre_release,
            no_audit_wheel=no_audit_wheel,
        )
        if isinstance(func, MarkedFunction):
            func.pythonpacks.append(pack_item)
            return func
        return MarkedFunction(func, pythonpacks=[pack_item])

    return func_wrapper


def with_running_options(
    *,
    engine: Optional[str] = None,
    cpu: Optional[int] = None,
    memory: Optional[Union[str, int]] = None,
    memory_limit: Optional[Union[str, int]] = None,
    gu: Optional[int] = None,
    gu_quota: Optional[Union[str, List[str]]] = None,
    gu_type: Optional[str] = None,
    **kwargs,
):
    """
    Select the execution engine and resources for a UDF.

    Parameters
    ----------
    engine : str, optional
        Engine name, such as ``"DPE"`` or ``"MCSQL"``. Case is ignored.
        If omitted, follow the job's engine selection. The engine must be
        available and support the operation.
    cpu : int, optional
        Positive CPU count for each task. Omitted fields inherit
        ``options.function.default_running_options``.
    memory : int or str, optional
        Starting memory for each task. Integers are in GiB; strings accept units
        such as ``"512MiB"`` or ``"8GiB"``. Omission inherits the shared default.
    memory_limit : int or str, optional
        Upper bound for elastic-memory retries of eligible DPE CPU UDFs, in the
        same units as ``memory``. The merged settings must contain ``memory``.
        Equal values disable expansion; a smaller limit is invalid. ``None``
        inherits a shared limit. Cluster resources can cap the effective limit.
    gu : int, optional
        Positive GPU-unit count for a DPE UDF. Cannot be combined with ``cpu``
        or ``memory``.
    gu_quota : str or list of str, optional
        GPU quota nickname, or nicknames in usage-priority order. Omission uses
        ``options.session.gu_quota_name``. A quota alone does not request a GPU.
    gu_type : str, optional
        Specification for a Spot GU quota, such as ``"gu.gu8tf-8x.spot"``.
        Requires ``gu_quota`` and cannot be combined with ``cpu`` or ``memory``.
        The specification determines the GPU count.
    **kwargs
        Additional running options. ``image_name`` delegates to
        :func:`with_image_options` to select a registered MaxCompute image.

    Returns
    -------
    callable
        Decorator that attaches engine and resource settings to a callable.

    Raises
    ------
    TypeError
        If ``memory`` or ``memory_limit`` is not an integer or string.
    ValueError
        If a numeric resource request is non-positive, ``gu_type`` has no quota,
        or a GPU request is combined with CPU or memory. Invalid memory sizes or
        limits also fail when the effective resource settings are collected.

    Notes
    -----
    Use one decorator for all engine and resource settings. When an outer
    ``with_running_options`` decorator specifies an engine or resource option,
    it replaces the inner engine and entire resource dictionary; fields are
    not merged between decorators. Missing fields then inherit
    ``options.function.default_running_options`` when the operation is built.
    An empty decorator, or one that only sets ``image_name``, leaves the
    existing engine and resource settings unchanged.

    CPU and memory apply to DPE and MCSQL. Elastic memory and GPU resources apply
    only to DPE. Resource requests do not make a Python loop parallel.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> from maxframe.udf import with_running_options
    >>> @with_running_options(engine="DPE", cpu=2, memory="8GiB")
    ... def double(batch):
    ...     return batch.assign(value=batch["value"] * 2)
    >>> df = md.DataFrame({"value": [1, 2]})
    >>> result = df.mf.apply_chunk(
    ...     double, output_type="dataframe", dtypes={"value": "int64"},
    ...     index=df.index, skip_infer=True,
    ... )
    >>> result.execute().fetch()  # doctest: +SKIP
       value
    0      2
    1      4
    """
    engine = engine.upper() if engine else None
    resources = kwargs.copy()

    if cpu is not None and isinstance(cpu, int) and cpu <= 0:
        raise ValueError("cpu must be greater than 0")
    if memory is not None:
        if not isinstance(memory, (int, str)):
            raise TypeError("memory must be an int or str")
        if isinstance(memory, int) and memory <= 0:
            raise ValueError("memory must be greater than 0")
    if memory_limit is not None:
        if not isinstance(memory_limit, (int, str)):
            raise TypeError("memory_limit must be an int or str")
        if isinstance(memory_limit, int) and memory_limit <= 0:
            raise ValueError("memory_limit must be greater than 0")
    if gu is not None and gu <= 0:
        raise ValueError("gu must be greater than 0")
    if gu_type is not None and not gu_quota:
        raise ValueError("gu_type requires gu_quota to be specified")
    use_gpu = gu_type is not None or is_positive_integer(gu)
    if use_gpu and (cpu or memory):
        raise ValueError("gu or gu_type can't be specified with cpu or memory")

    if cpu:
        resources["cpu"] = cpu
    if memory:
        resources["memory"] = memory
    if memory_limit:
        resources["memory_limit"] = memory_limit

    if isinstance(gu_quota, str):
        gu_quota = [gu_quota]

    # Only set gpu/gu_quota if they have values, avoid setting None
    if gu is not None:
        resources["gpu"] = gu
    if gu_quota is not None:
        resources["gu_quota"] = gu_quota
    if gu_type is not None:
        resources["gu_type"] = gu_type

    def func_wrapper(func):
        if not all(
            v is None for v in (engine, cpu, memory, gu, gu_quota, memory_limit)
        ):
            if isinstance(func, MarkedFunction):
                func.expect_engine = engine
                func.expect_resources = resources
                func.gpu = use_gpu
            else:
                func = MarkedFunction(
                    func,
                    expect_engine=engine,
                    expect_resources=resources,
                    gpu=use_gpu,
                )
        # Delegate other settings
        if "image_name" in kwargs:
            func = with_image_options(**kwargs)(func)
        return func

    return func_wrapper


def with_image_options(
    image_name: Optional[str] = None,
    **kwargs,
):
    """
    Select a registered custom image for a DPE UDF.

    Parameters
    ----------
    image_name : str
        Name of an image already registered in MaxCompute. Required despite
        the ``None`` default in the signature.
    **kwargs
        Additional image options passed to the runtime.

    Returns
    -------
    callable
        Decorator that attaches image settings to a callable.

    Raises
    ------
    ValueError
        If ``image_name`` is empty or the callable already has image settings.

    Notes
    -----
    This decorator does not select DPE automatically. Use
    :func:`with_running_options` to select it. The function-level image replaces
    ``options.session.image_name``.

    Examples
    --------
    >>> from maxframe.udf import with_image_options, with_running_options
    >>> @with_running_options(engine="DPE")
    ... @with_image_options("registered_image_name")
    ... def score(batch):
    ...     import custom_scoring_pkg
    ...     return batch.assign(score=custom_scoring_pkg.score_batch(batch["text"]))
    """
    if not image_name:
        raise ValueError("image_name is required")

    image_options_config = {"name": image_name, **kwargs}

    def func_wrapper(func):
        if isinstance(func, MarkedFunction):
            existing = func.image_options
            if existing:
                raise ValueError(
                    f"Function image_options already set to '{existing.get('name')}', "
                    f"cannot set to '{image_name}' again"
                )
            func.image_options = image_options_config
            return func
        return MarkedFunction(
            func,
            image_options=image_options_config,
        )

    return func_wrapper


def with_network_options(
    *,
    vpc_network_link: Optional[str] = None,
    public_whitelist: Optional[List[str]] = None,
    internal_whitelist: Optional[List[str]] = None,
):
    """
    Configure public, internal, or VPC access for a DPE UDF.

    Parameters
    ----------
    vpc_network_link : str, optional
        Name of a MaxCompute network link for a VPC service. Create it using the
        `Access VPC solution <https://help.aliyun.com/zh/maxcompute/user-guide/access-vpc-solution-direct-connection>`_.
    public_whitelist : list of str, optional
        Public IP addresses or domains. These must be approved for the project
        through a service ticket before use.
    internal_whitelist : list of str, optional
        Alibaba Cloud internal IP addresses or domains, such as an OSS internal
        endpoint. Obtain each endpoint from the corresponding service's docs.

    Returns
    -------
    callable
        Decorator that attaches network settings to a callable.

    Raises
    ------
    ValueError
        If repeated decorators request different VPC network links.

    Notes
    -----
    Public whitelists are combined, as are internal whitelists, across decorators
    and DPE shared settings. Only one VPC link is allowed per function.
    The decorator does not select DPE automatically.

    Examples
    --------
    >>> from maxframe.udf import with_network_options, with_running_options
    >>> @with_running_options(engine="DPE")
    ... @with_network_options(vpc_network_link="configured_network_link")
    ... def score(batch):
    ...     import scoring_client
    ...     return batch.assign(score=scoring_client.score_batch(batch["text"]))
    """

    def func_wrapper(func):
        if isinstance(func, MarkedFunction):
            pre_public_whitelist_set = (
                set(func.public_network_whitelist)
                if func.public_network_whitelist
                else set()
            )
            curr_public_whitelisted_set = (
                set(public_whitelist) if public_whitelist else set()
            )
            func.public_network_whitelist = list(
                pre_public_whitelist_set | curr_public_whitelisted_set
            )

            pre_internal_whitelist_set = (
                set(func.internal_network_whitelist)
                if func.internal_network_whitelist
                else set()
            )
            curr_internal_whitelist_set = (
                set(internal_whitelist) if internal_whitelist else set()
            )
            func.internal_network_whitelist = list(
                pre_internal_whitelist_set | curr_internal_whitelist_set
            )

            if vpc_network_link:
                if func.vpc_network_link and func.vpc_network_link != vpc_network_link:
                    raise ValueError(
                        "multiple VPC network connections are not supported"
                    )
                func.vpc_network_link = vpc_network_link

            return func

        marked_func = MarkedFunction(func)
        marked_func.public_network_whitelist = public_whitelist
        marked_func.internal_network_whitelist = internal_whitelist
        marked_func.vpc_network_link = vpc_network_link
        return marked_func

    return func_wrapper


StorageOptions = Optional[Dict[str, Any]]


def with_fs_mount(
    path: str,
    mount_path: str,
    storage_options: StorageOptions = None,
):
    """
    Mount an OSS path for a DPE UDF.

    Parameters
    ----------
    path : str
        OSS URI including endpoint, bucket, and optional prefix, for example
        ``"oss://oss-cn-hangzhou.aliyuncs.com/bucket/models"``.
    mount_path : str
        Local path through which the UDF reads the mounted files.
    storage_options : dict
        Authentication settings. Provide ``role_arn``, or both
        ``access_key_id`` and ``access_key_secret``. Credentials are required;
        the default ``None`` does not provide anonymous access.

    Returns
    -------
    callable
        Decorator that appends a mount to the callable's mount settings.

    Raises
    ------
    ValueError
        If either path is missing or is not a string, authentication is missing,
        or mounts on the same callable use different ``storage_options``.

    Notes
    -----
    This decorator requires DPE but does not select the engine automatically.
    For several mounts on one function, use identical authentication settings.

    Examples
    --------
    >>> from pathlib import Path
    >>> from maxframe.udf import with_fs_mount, with_running_options
    >>> @with_running_options(engine="DPE")
    ... @with_fs_mount(
    ...     "oss://oss-cn-hangzhou.aliyuncs.com/bucket/models", "/mnt/models",
    ...     storage_options={"role_arn": "acs:ram::<account-id>:role/<role-name>"},
    ... )
    ... def model_size(batch):
    ...     size = Path("/mnt/models/model.bin").stat().st_size
    ...     return batch.assign(model_bytes=size)
    """
    mount_options = FsMountOptions(
        path=path,
        mount_path=mount_path,
        storage_options=storage_options or {},
    )
    mount_options.validate()

    def func_wrapper(func):
        if isinstance(func, MarkedFunction):
            # Validate that all mounts use identical storage_options
            if func.fs_mount:
                existing_opts = func.fs_mount[0].storage_options
                new_opts = mount_options.storage_options
                if existing_opts != new_opts:
                    raise ValueError(
                        f"All fs_mount decorators must use identical storage_options. "
                        f"Existing: {existing_opts}, New: {new_opts}"
                    )

            func.fs_mount.append(mount_options)
            return func

        return MarkedFunction(func, fs_mount=[mount_options])

    return func_wrapper


def get_udf_fs_mount(func: Callable) -> List[FsMountOptions]:
    return getattr(func, "fs_mount", None) or []


with_resource_libraries = with_resources


def get_udf_resources(func: Callable) -> List[Union[ODPSResourceObj, str]]:
    return getattr(func, "file_resources", None) or []


def get_udf_pythonpacks(func: Callable) -> List[PythonPackOptions]:
    return getattr(func, "pythonpacks", None) or []


def discover_marked_functions(func: Callable) -> List[MarkedFunction]:
    """
    Discover all MarkedFunction instances referenced in a function.
    """
    discovered: Set[MarkedFunction] = set()
    visited: set = set()

    def _discover(obj: Any) -> None:
        """Recursively discover MarkedFunction instances in an object."""
        # Avoid infinite recursion
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        # Handle functools.partial - check its func, args, and keywords
        if isinstance(obj, functools.partial):
            _discover(obj.func)
            for arg in obj.args:
                _discover(arg)
            for kwarg in obj.keywords.values() if obj.keywords else []:
                _discover(kwarg)
            return

        obj = unwrap_function(
            obj, stop_predicate=lambda x: isinstance(x, MarkedFunction)
        )

        # Check if it's a MarkedFunction
        if isinstance(obj, MarkedFunction):
            if obj not in discovered:
                discovered.add(obj)
            # Also check the inner function of MarkedFunction
            inner_func = obj.func
            if inner_func is not None and inner_func is not obj:
                _discover(inner_func)
            return

        # If it's a callable, inspect its closures and globals
        if callable(obj) and hasattr(obj, "__code__"):
            code = obj.__code__

            # Inspect closure variables
            if hasattr(obj, "__closure__") and obj.__closure__:
                for cell in obj.__closure__:
                    try:
                        cell_contents = cell.cell_contents
                        _discover(cell_contents)
                    except ValueError:
                        # Cell is empty
                        pass

            # Inspect global variables referenced by the function
            if hasattr(obj, "__globals__"):
                for name in code.co_names:
                    if name in obj.__globals__:
                        _discover(obj.__globals__[name])

            # Also check co_consts for nested functions/lambdas
            for const in code.co_consts:
                if callable(const) and const is not obj:
                    _discover(const)

    _discover(func)
    return list(discovered)


def copy_func_scheduling_hints(func: Callable, op: Any) -> None:
    expect_engine = None
    expect_gpu = None
    fs_mount = None
    image_options = None
    default_options = options.function.default_running_options or {}

    marked_funcs = discover_marked_functions(func)
    if len(marked_funcs) > 1:
        raise ValueError(
            "Multiple functions are marked with running options, "
            "please keep only one of them"
        )
    elif marked_funcs:
        marked_func = marked_funcs[0]
        # copy from marked function
        expect_engine = marked_func.expect_engine
        expect_resources = (marked_func.expect_resources or {}).copy()
        expect_gpu = marked_func.gpu
        fs_mount = marked_func.fs_mount
        image_options = marked_func.image_options

        # merge default options if not set
        for key, value in default_options.items():
            if key not in expect_resources or expect_resources.get(key) is None:
                expect_resources[key] = value
    else:
        # copy from default options
        expect_resources = default_options.copy()

    memory_limit = expect_resources.get("memory_limit")
    if memory_limit is not None:
        memory = expect_resources.get("memory")
        if memory is None:
            raise ValueError("memory must be specified when memory_limit is set")
        memory_mb = parse_size_to_megabytes(memory)
        memory_limit_mb = parse_size_to_megabytes(memory_limit)
        if memory_limit_mb < memory_mb:
            raise ValueError("memory_limit must be greater than or equal to memory")
        if memory_limit_mb == memory_mb:
            expect_resources.pop("memory_limit")

    # Validate and adjust resource ratio constraints on client side
    expect_resources, _ = validate_and_adjust_resource_ratio(
        expect_resources,
        max_memory_cpu_ratio=options.function.allowed_max_memory_cpu_ratio,
        adjust=True,
    )

    # If GPU is required but gu_quota not set, inherit from global setting
    if expect_resources.get("gpu"):
        expect_resources["gu_quota"] = expect_resources.get(
            "gu_quota", [options.session.gu_quota_name]
        )

    # Use session default image if not set
    if not image_options and options.session.image_name:
        image_options = {"name": options.session.image_name}

    if expect_engine:
        op.expect_engine = expect_engine
    if expect_resources:
        op.expect_resources = expect_resources
    if expect_gpu:
        op.gpu = expect_gpu
    if fs_mount:
        op.fs_mount = fs_mount
    if image_options:
        op.image_options = image_options
