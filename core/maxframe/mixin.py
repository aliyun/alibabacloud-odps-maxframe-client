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

import logging.config
from typing import Union

import traitlets as T
from tornado.log import enable_pretty_logging
from traitlets.config import Configurable

from . import env
from .utils import trait_from_env


class ServiceConfigMixin(Configurable):
    port_env = env.MAXFRAME_SERVICE_PORT
    port = T.Integer(
        7953,
        config=True,
        help=f"Port on which to listen ({port_env} env var)",
    )
    port_default = trait_from_env("port", port_env)

    port_retries_env = env.MAXFRAME_SERVICE_PORT_RETRIES
    port_retries = T.Integer(
        50,
        config=True,
        help=f"""Number of ports to try if the specified port is not available
                           ({port_retries_env} env var)""",
    )
    port_retries_default = trait_from_env("port_retries", port_retries_env)

    ip_env = env.MAXFRAME_SERVICE_LISTEN_ADDRESS
    ip = T.Unicode(
        "127.0.0.1",
        config=True,
        help=f"IP address on which to listen ({ip_env} env var)",
    )
    ip_default = trait_from_env("ip", ip_env)

    allow_origin_env = env.MAXFRAME_SERVICE_ALLOW_ORIGIN
    allow_origin = T.Unicode(
        "",
        config=True,
        help=f"Sets the Access-Control-Allow-Origin header. ({allow_origin_env} env var)",
    )
    allow_origin_default = trait_from_env("allow_origin", allow_origin_env)

    # Base URL
    base_url_env = env.MAXFRAME_SERVICE_BASE_URL
    base_url = T.Unicode(
        "/",
        config=True,
        help=f"The misc path for mounting all API resources ({base_url_env} env var)",
    )
    base_url_default = trait_from_env("base_url", base_url_env)


class LoggerConfigMixin(Configurable):
    log_config_file_env = env.MAXFRAME_SERVICE_LOG_CONFIG_FILE
    log_config_file = T.Unicode(
        "",
        config=True,
        help=f"Sets the config file of logger. ({log_config_file_env} env var)",
    )
    log_config_file_default = trait_from_env("log_config_file", log_config_file_env)

    def init_logger(self, log_level: Union[int, str] = None):
        """
        Init the logger. If the environment variable MAXFRAME_SERVICE_LOG_CONFIG_FILE is
        set, the logger will be configured by this config file.
        If log_level is set, it will override the level defined in the config file.

        Parameters
        ----------
        log_level : int or str
            The log level.
        """
        # Enable the same pretty logging the server uses
        enable_pretty_logging()
        if self.log_config_file:
            logging.config.fileConfig(
                self.log_config_file, disable_existing_loggers=False
            )
        # LogLevel from command line or environment has a high priority than config file.
        if log_level:
            logging.getLogger().setLevel(log_level)
        # Adjust kubernetes logging level to hide secrets when logging
        logging.getLogger("kubernetes").setLevel(logging.WARNING)
