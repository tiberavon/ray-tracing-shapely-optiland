"""
Python-specific module: test version
Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The version the package reports is the one it is released as: __version__
equals pyproject.toml's, which is what PyPI, the tag and the citation file
carry. Skipped where the source tree is not there (an installed wheel).
"""

import tomllib
from pathlib import Path

import pytest

import ray_tracing_shapely_optiland

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.mark.skipif(not PYPROJECT.exists(), reason="no source tree: pyproject.toml is not installed")
def test_the_reported_version_is_the_released_one():
    with open(PYPROJECT, "rb") as f:
        assert ray_tracing_shapely_optiland.__version__ == tomllib.load(f)["project"]["version"]
