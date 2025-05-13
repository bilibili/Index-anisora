
# Copyright (c) Ming Ding, et al. in KEG, Tsinghua University.
#
# LICENSE file in the root directory of this source tree.

import json
import sys
import os
from pathlib import Path

from setuptools import find_packages, setup


def _requirements():
    return Path("requirements.txt").read_text()

setup(
    name="SwissArmyTransformer",
    version='0.4.12', # don't forget to modify __init__.py __version__
    description="A transformer-based framework with finetuning as the first class citizen.",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    install_requires=_requirements(),
    entry_points={},
    packages=find_packages(),
    url="https://github.com/THUDM/SwissArmyTransformer",
    author="Ming Ding, et al.",
    author_email="dm_thu@qq.com",
    scripts={},
    include_package_data=True,
    python_requires=">=3.5",
    license="Apache 2.0 license"
)
