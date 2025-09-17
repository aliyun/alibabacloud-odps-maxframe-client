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

import os
import shutil

from setuptools import Extension, setup

pack_root = os.path.dirname(os.path.abspath(__file__))

# load cythonize_all from _cythonize_all.py
with open(os.path.join(pack_root, "_cythonize_all.py"), "r") as cy_src:
    _locals = locals()
    exec(cy_src.read(), globals(), _locals)
    cythonize_all_pyx = _locals["cythonize_all_pyx"]

# The pyx with C sources.
ext_include_source_map = {
    "maxframe/_utils.pyx": [
        ["maxframe/lib/mmh3_src"],
        ["maxframe/lib/mmh3_src/MurmurHash3.cpp"],
    ],
}

extensions = cythonize_all_pyx(pack_root, ext_include_source_map) + [
    Extension(
        "maxframe.lib.mmh3",
        [
            "maxframe/lib/mmh3_src/mmh3module.cpp",
            "maxframe/lib/mmh3_src/MurmurHash3.cpp",
        ],
    )
]

long_description = None
readme_files = [
    f"{pack_root}/../README.rst",
    f"{pack_root}/../README.md",
    f"{pack_root}/README.rst",
    f"{pack_root}/README.md",
]
for readme_file in readme_files:
    if os.path.exists(readme_file):
        with open(readme_file) as f:
            long_description = f.read()
        break


setup_options = dict(
    long_description=long_description or "",
    ext_modules=extensions,
)
try:
    if os.path.exists(f"{pack_root}/../README.rst"):
        shutil.copy(f"{pack_root}/../README.rst", f"{pack_root}/README.rst")
    setup(**setup_options)
finally:
    if os.path.exists(f"{pack_root}/README.rst"):
        os.unlink(f"{pack_root}/README.rst")
