# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import contextlib
import contextvars
import traceback
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Union

from ..utils import get_python_tag
from .validators import (
    ValidatorType,
    all_validator,
    any_validator,
    is_bool,
    is_dict,
    is_in,
    is_integer,
    is_null,
    is_numeric,
    is_string,
)

_DEFAULT_REDIRECT_WARN = "Option {source} has been replaced by {target} and might be removed in a future release."
_DEFAULT_MAX_ALIVE_SECONDS = 3 * 24 * 3600
_DEFAULT_MAX_IDLE_SECONDS = 3600
_DEFAULT_SPE_OPERATION_TIMEOUT_SECONDS = 120
_DEFAULT_UPLOAD_BATCH_SIZE = 4096
_DEFAULT_TEMP_LIFECYCLE = 1
_DEFAULT_TASK_START_TIMEOUT = 60


class OptionError(Exception):
    pass


class Redirection:
    def __init__(self, item: str, warn: Optional[str] = None):
        self._items = item.split(".")
        self._warn = warn
        self._warned = True
        self._parent = None

    def bind(self, attr_dict):
        self._parent = attr_dict
        self.getvalue()
        self._warned = False

    def getvalue(self, silent: bool = False) -> Any:
        if not silent and self._warn and not self._warned:
            in_completer = any(
                1 for st in traceback.extract_stack() if "completer" in st[0].lower()
            )
            if not in_completer:
                self._warned = True
                warnings.warn(self._warn)
        conf = self._parent.root
        for it in self._items:
            conf = getattr(conf, it)
        return conf

    def setvalue(self, value: str, silent: bool = False) -> None:
        if not silent and self._warn and not self._warned:
            self._warned = True
            warnings.warn(self._warn)
        conf = self._parent.root
        for it in self._items[:-1]:
            conf = getattr(conf, it)
        setattr(conf, self._items[-1], value)


class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        self._inited = False
        self._parent = kwargs.pop("_parent", None)
        self._root = None
        super().__init__(*args, **kwargs)
        self._inited = True

    @property
    def root(self):
        if self._root is not None:
            return self._root
        if self._parent is None:
            self._root = self
        else:
            self._root = self._parent.root
        return self._root

    def __getattr__(self, item: str):
        if item in self:
            val = self[item]
            if isinstance(val, AttributeDict):
                return val
            elif isinstance(val[0], Redirection):
                return val[0].getvalue()
            else:
                return val[0]
        return object.__getattribute__(self, item)

    def __dir__(self):
        return list(self.keys())

    def register(
        self, key: str, value: Any, validator: Optional[ValidatorType] = None
    ) -> None:
        self[key] = value, validator
        if isinstance(value, Redirection):
            value.bind(self)

    def unregister(self, key: str) -> None:
        del self[key]

    def add_validator(self, key: str, validator: ValidatorType) -> None:
        value, old_validator = self[key]
        validators = getattr(
            old_validator,
            "validators",
            [old_validator] if callable(old_validator) else [],
        )
        validators.append(validator)
        self[key] = (value, all_validator(*validators))

    def _setattr(self, key: str, value: Any, silent: bool = False) -> None:
        if not silent and key not in self:
            raise OptionError(f"Cannot identify configuration name '{key}'.")

        if not isinstance(value, AttributeDict):
            validate = None
            if key in self:
                val = self[key]
                validate = self[key][1]
                if validate is not None:
                    if not validate(value):
                        raise ValueError(f"Cannot set value {value}")
                if isinstance(val[0], Redirection):
                    val[0].setvalue(value)
                else:
                    self[key] = value, validate
            else:
                self[key] = value, validate
        else:
            self[key] = value

    def __setattr__(self, key: str, value: Any):
        if key == "_inited":
            super().__setattr__(key, value)
            return
        try:
            object.__getattribute__(self, key)
            super().__setattr__(key, value)
            return
        except AttributeError:
            pass

        if not self._inited:
            super().__setattr__(key, value)
        else:
            self._setattr(key, value)

    def to_dict(self) -> Dict[str, Any]:
        result_dict = dict()
        for k, v in self.items():
            if isinstance(v, AttributeDict):
                result_dict.update((f"{k}.{sk}", sv) for sk, sv in v.to_dict().items())
            elif isinstance(v[0], Redirection):
                continue
            else:
                result_dict[k] = v[0]
        return result_dict


class Config:
    def __init__(self, config=None):
        self._config = config or AttributeDict()
        self._remote_options = set()

    def __dir__(self):
        return list(self._config.keys())

    def __getattr__(self, item: str):
        return getattr(self._config, item)

    def __setattr__(self, key: str, value: Any):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        setattr(self._config, key, value)

    def register_option(
        self,
        option: str,
        value: Any,
        validator: Optional[ValidatorType] = None,
        remote: bool = False,
    ) -> None:
        assert validator is None or callable(validator)
        splits = option.split(".")
        conf = self._config

        for name in splits[:-1]:
            config = conf.get(name)
            if config is None:
                val = AttributeDict(_parent=conf)
                conf[name] = val
                conf = val
            elif not isinstance(config, dict):
                raise AttributeError(
                    f"Fail to set option: {option}, conflict has encountered"
                )
            else:
                conf = config

        key = splits[-1]
        if conf.get(key) is not None:
            raise AttributeError(f"Fail to set option: {option}, option has been set")

        conf.register(key, value, validator)
        if remote:
            self._remote_options.add(option)

    def redirect_option(
        self, option: str, target: str, warn: str = _DEFAULT_REDIRECT_WARN
    ) -> None:
        redir = Redirection(target, warn=warn.format(source=option, target=target))
        self.register_option(option, redir)

    def unregister_option(self, option: str) -> None:
        splits = option.split(".")
        conf = self._config
        for name in splits[:-1]:
            config = conf.get(name)
            if not isinstance(config, dict):
                raise AttributeError(
                    f"Fail to unregister option: {option}, conflict has encountered"
                )
            else:
                conf = config

        key = splits[-1]
        if key not in conf:
            raise AttributeError(
                f"Option {option} not configured, thus failed to unregister."
            )
        conf.unregister(key)

    def update(self, new_config: Union["Config", Dict[str, Any]]) -> None:
        if not isinstance(new_config, dict):
            new_config = new_config._config
        for option, value in new_config.items():
            try:
                self.register_option(option, value)
            except AttributeError:
                attrs = option.split(".")
                cur_cfg = self
                for sub_cfg_name in attrs[:-1]:
                    cur_cfg = getattr(cur_cfg, sub_cfg_name)
                setattr(cur_cfg, attrs[-1], value)

    def add_validator(self, option: str, validator: ValidatorType) -> None:
        splits = option.split(".")
        conf = self._config
        for name in splits[:-1]:
            config = conf.get(name)
            if not isinstance(config, dict):
                raise AttributeError(
                    f"Fail to add validator: {option}, conflict has encountered"
                )
            else:
                conf = config

        key = splits[-1]
        if key not in conf:
            raise AttributeError(
                f"Option {option} not configured, thus failed to set validator."
            )
        conf.add_validator(key, validator)

    def to_dict(self, remote_only: bool = False) -> Dict[str, Any]:
        res = self._config.to_dict()
        if not remote_only:
            return res
        return {k: v for k, v in res.items() if k in self._remote_options}


default_options = Config()

default_options.register_option(
    "execution_mode", "trigger", validator=is_in(["trigger", "eager"])
)
default_options.register_option(
    "python_tag", get_python_tag(), validator=is_string, remote=True
)
default_options.register_option(
    "client.task_start_timeout", _DEFAULT_TASK_START_TIMEOUT, validator=is_integer
)
default_options.register_option("sql.enable_mcqa", True, validator=is_bool, remote=True)
default_options.register_option(
    "sql.generate_comments", True, validator=is_bool, remote=True
)
default_options.register_option("sql.settings", {}, validator=is_dict, remote=True)

default_options.register_option(
    "session.max_alive_seconds",
    _DEFAULT_MAX_ALIVE_SECONDS,
    validator=is_numeric,
    remote=True,
)
default_options.register_option(
    "session.max_idle_seconds",
    _DEFAULT_MAX_IDLE_SECONDS,
    validator=is_numeric,
    remote=True,
)
default_options.register_option(
    "session.upload_batch_size",
    _DEFAULT_UPLOAD_BATCH_SIZE,
    validator=is_integer,
)
default_options.register_option(
    "session.table_lifecycle",
    None,
    validator=any_validator(is_null, is_integer),
)
default_options.register_option(
    "session.temp_table_lifecycle",
    _DEFAULT_TEMP_LIFECYCLE,
    validator=is_integer,
    remote=True,
)

default_options.register_option("warn_duplicated_execution", False, validator=is_bool)
default_options.register_option("dataframe.use_arrow_dtype", True, validator=is_bool)
default_options.register_option(
    "dataframe.arrow_array.pandas_only", True, validator=is_bool
)
default_options.register_option(
    "optimize.head_optimize_threshold", 1000, validator=is_integer
)
default_options.register_option(
    "show_progress", "auto", validator=any_validator(is_bool, is_string)
)

################
# SPE Settings #
################
default_options.register_option(
    "spe.operation_timeout_seconds",
    _DEFAULT_SPE_OPERATION_TIMEOUT_SECONDS,
    validator=is_integer,
    remote=True,
)

default_options.register_option(
    "spe.task.settings", dict(), validator=is_dict, remote=True
)

_options_ctx_var = contextvars.ContextVar("_options_ctx_var")


def reset_global_options():
    global _options_ctx_var

    _options_ctx_var = contextvars.ContextVar("_options_ctx_var")
    _options_ctx_var.set(default_options)


reset_global_options()


def get_global_options(copy: bool = False) -> Config:
    ret = _options_ctx_var.get(None)

    if ret is None:
        if not copy:
            ret = default_options
        else:
            ret = Config(deepcopy(default_options._config))
        _options_ctx_var.set(ret)
    return ret


def set_global_options(opts: Config) -> None:
    _options_ctx_var.set(opts)


@contextlib.contextmanager
def option_context(config: Dict[str, Any] = None):
    global_options = get_global_options(copy=True)

    try:
        config = config or dict()
        local_options = Config(deepcopy(global_options._config))
        local_options.update(config)
        set_global_options(local_options)
        yield local_options
    finally:
        set_global_options(global_options)


class OptionsProxy:
    def __dir__(self):
        return dir(get_global_options())

    def __getattribute__(self, attr):
        return getattr(get_global_options(), attr)

    def __setattr__(self, key, value):
        setattr(get_global_options(), key, value)


options = OptionsProxy()
