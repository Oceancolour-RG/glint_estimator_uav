#!/usr/bin/env python3

"""
setup for image registration module
"""

import pathlib
from setuptools import setup


HERE = pathlib.Path(__file__).parent.absolute()
PKGDIR = HERE / "src"
README = (HERE / "README.md").read_text()

# Parse the version from the main __init__.py
with open(PKGDIR / "__init__.py") as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

setup(
    name="glint_estimator_uav",
    version=version,
    description="Estimates location and spread of sunglint contamination in UAV imagery",
    long_description=README,
    author="Rodrigo A. Garcia",
    author_email="rodrigo.garcia@uwa.edu.au",
    python_requires=">=3.8, <3.12",
    packages=["glint_estimator_uav"],
    package_dir={"glint_estimator_uav": "src"},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.23.5",
        "numexpr>=2.8.4",
        "pysolar>=0.10",
        "matplotlib>=3.7.0"
    ],
)

