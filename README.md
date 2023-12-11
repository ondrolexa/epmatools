# Welcome to EPMAtools

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/epmatools/badge/)](https://epmatools.readthedocs.io/)
[![codecov](https://codecov.io/gh/ondrolexa/epmatools/graph/badge.svg?token=KTY1CYPJSF)](https://codecov.io/gh/ondrolexa/epmatools)

Python module to manipulate EPMA analyses

## Installation

The Python package `epmatools` can be installed from PyPI:

```
python -m pip install epmatools
```

### Installation of master version using conda/mamba

You can also use the provided conda/mamba environment file to install it. Download the package and unzip. Open terminal in unpacked folder and create environment:

```
conda env create -f environment.yml
```

then activate newly created environment:

```
mamba activate epmatools
```

and install `epmatools` package using pip:

```
# pip install .
```

### Development installation

If you want to contribute to the development of `epmatools`, we recommend
the editable installation from this repository:

```
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Usage

Check example in the following [jupyter notebook](https://nbviewer.org/github/ondrolexa/epmatools/blob/main/notebooks/demo.ipynb)

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
