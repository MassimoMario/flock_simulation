<div align="center">

# What the Flock! :bird: :bird: :bird:

</div>

# Flock simulation
Agent based imulation of Flock dynamic with Python. The scripts contained in this repository allow to simulate the flock behaviour given some basic rules, and visualizes it with an animation. This project also includes unit tests to ensure the correct functionality of the simulation.

# Table of Contents
1. [Installation](#Installation)
2. [Requirements](#Requirements)
3. [Repository structure](#Repository-structure)
4. [Documentation](#Documentation)
5. [Scripts pverview](#Scripts-overview)
6. [Usage and examples][(#Usage-and-examples)
7. [Theory background](#Theory-background)

# Installation
To start using the repository, irst clone it:

```
git clone https://github.com/MassimoMario/flock_simulation.git
```

# Requirements
This project requires **Python &ge; 3.8** and the following libraries:
- `numpy`
- `matplotlib`
- `ipython`
- `tqdm`
- `argpare`
- `configparser`
- `pytest`
- `pytest-cov`

# Repository structure
The repository contains the following folders and files:
- [`simulation.py`](simulation.py) is the main script for simulating and animating the flock.
- [`scripts`](scripts) folder containing the flock class, the test script and the utils scripts
  - [`flock_class.py`](scripts/flock_class.py) contains the class Flock definition, where all the simulation computation are described
  - [`test.py`](scripts/test.py) is the unit test scripts made using pytest
  - [`utils.py`](scripts/utils) script contains the animation function and some useful function for the command line interface arguments interpretation
- [`config`](config) folder contains 3 configuration files `.ini` to be used from command line as inputs in `simulation.py` script

# Documentation

# Scripts overview

# Usage and examples

# Theory background
