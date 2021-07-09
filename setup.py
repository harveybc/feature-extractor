# -*- coding: utf-8 -*-
"""
    Setup file for heuristic_ts.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.3.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import sys

from pkg_resources import VersionConflict, require
from setuptools import setup
from setuptools import find_packages

# sys.path.append('data_logger\\')
# from data_logger.heuristic_ts.heuristic_ts import HeuristicTS

try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    setup(
        packages=find_packages(exclude=["contrib", "docs", "tests"]),
        use_pyscaffold=True,
    )
