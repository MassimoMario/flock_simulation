<div align="center">

# :bird: :bird: :bird: What the Flock! :bird: :bird: :bird:

</div>

# Flock simulation
Agent based imulation of Flock dynamic with Python. The scripts contained in this repository allow to simulate the flock behaviour given some basic rules, and visualizes it with an animation. This project also includes unit tests to ensure the correct functionality of the simulation.

# Table of Contents
1. [Installation](#Installation)
2. [Requirements](#Requirements)
3. [Repository structure](#Repository-structure)
4. [Documentation](#Documentation)
5. [Scripts overview](#Scripts-overview)
6. [Usage and examples](#Usage-and-examples)
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
- `pytest`
- `pytest-cov`

To install them you can run on the Bash:

```
pip install -r requirements.txt
```

# Repository structure
The repository contains the following folders and files:
- [`simulation.py`](simulation.py) is the main script for simulating and animating the flock.
- [`scripts`](scripts) folder containing the flock class, the test script and the utils scripts
  - [`flock_class.py`](scripts/flock_class.py) contains the class Flock definition, where all the simulation computation are described
  - [`test.py`](scripts/test.py) is the unit test scripts made using pytest
  - [`utils.py`](scripts/utils.py) script contains the animation function and some useful functions for the configparser arguments interpretation
- [`config`](config) folder contains 3 configuration files `.ini` to be used from command line as inputs in `simulation.py` script
- [`requirements.txt`](requirements.txt) file contains the list of dependencies required for the project
- [`docs`](docs) folder contains the source files used to generate the documentation.

# Documentation
The documentation can be seen at this [link](https://massimomario.github.io/flock_simulation/).

It's a Github page containing the html files generated using **Sphinx** starting from docstrings in the code.

[`docs`](docs) folder contains the source files used to generate this documentation.

# Scripts overview
- [`scripts/flock_class.py`](scripts/flock_class.py)

Script containing the Flock class definition. Each Flock object is an ensamble of N birds and can simulate their interaction and dynamic. The only methods accessible to the user are `init_given_position`, `init_given_velocities` which initialize the birds positions/velocities with given arrays, and `simulate` which simulate the flock dynamic given physic parameters such as separation, alignment, coherence and avoidance (see [Theory background](#Theory-background) section).

---

- [`scripts/utils.py`](scripts/utils.py)

Script containing the `animation` function, which animates and displays the flock dynamic with `matplotlib.animation` and `IPython.display` taking as input the resulting arrays of a Flock object simulation method. The script contains also some useful functions for the configparser arguments interpretation, which cast a string in the corresponding correct type in order to give it as input to the simulation.

---

- :wrench: [`scripts/test.py`](script/test.py)

Script containing unit tests for all methods in [`flock_class.py`](script/flock_class.py). The tests ensure correctness and stability of the code following the unit testing approach and are implemented using `pytest`.

**Usage** : To run all the tests, navigate in the `scripts` folder and run the test script:
	
```
cd scripts
pytest test.py
```

**Coverage** : To check the test coverage, use `pytest-cov`:

```
cd scripts
pytest --cov=flock_class test.py
```
	
---


- [`simulation.py`](simulation.py)

It's the main script where all the required parameters are taken from command line using `argparse` or from a config file `.ini` using `configparser` library. In this script a flock dynamic is simulated and animated using a Flock object and the animation function from [`utils.py`](scripts/utils.py). The user can choose to save the animation if GIF format using the `--save` argument in the command line.

The simulation is agent-based, it computes the interaction between birds and update their positions and velocities following some simple rules. The main paramaters of the simulation are **separation**, **alignment**, **coherence** and **avoidance**. Each of those parameters controls the strength of a force acting on each bird. See [Theory background](#Theory-background) section for a mathematical description.

:warning: If you are providing a config file, take in mind that explicit command line arguments overwrite the config file ones.

# Usage and examples
The main script [`simulation.py`](simulation.py) can be runned from the command line providing different argument ending in different configuration.

## :information_source: Help
For a complete list of parameters and their descriptions, run:

```
python simulation.py --help
```

## :one: Simulate and animate a flock
Example of a simulation with a parameters that produce a good behaviour flock dynamic:

```
python simulation.py --N 300 --separation 11 --alignment 2.35 --coherence 1.1 --visual_range 30 --avoid_range 23 
```

![Example GIF of a good behaviour flock dynamic](images_gif/flock_simulation.gif)

## :two: Provide intial positions and velocities
You can provide initial positions and velocities of N birds:

```
python simulation.py --N 3 --positions_i 1 1 --positions_i 50 50 --positions_i 30 55 --velocities_i 1 1 --velocities_i -5 5 --velocities_i -1.1 -2.3
```
:warning: The previous setup produce an initial position array equal to `[[1,1],[50,50],[30,55]]` and an initial velocity array equal to `[[1,1],[-5,5],[-1.1,-2.3]]`

## :three: Provide a config file
You can provide simulation parameter from a `.ini` config file:

```
python simulation.py --config config/config_good_behaviour.ini
```

## :four: Save the animation
You can save the animation as a GIF:

```
python simulation.py --save True
```
# Theory background
To simulate the collective emerging behaviour of a flock four simple rules have been taken into account that each bird must follow:

- **Separation**: move away from neighbour that is too near to avoid collision
- **Alignment**: adapt its velocity to the flock average velocity to move in the same direction as other birds
- **Coherence**: adapt its position to the flock average position moving toward the center of the flock to keep group toghether
- **Avoidance**: move away from obstacles/simulation barriers

![Graphical representation of flock rules](images_gif/flocking_rules.png)
:--:
*Graphical representation of flock rules. From [DataSctientest](https://datascientest.com/en/the-flocking-algorithm-or-the-simulation-of-collective-behaviour)*

Mathematically, each rule is associated with a force and a scaling factor, and the simulation update is done using a pseudo accerelerated system formula.

First the flock mean position and velocity is computed, ![Formula](https://latex.codecogs.com/png.latex?\vec{x}) and ![Formula](https://latex.codecogs.com/png.latex?\vec{v}). Then for each bird $i$ with position ![Formula](https://latex.codecogs.com/png.latex?\vec{x}_i) is computed the  position of the nearest bird to ![Formula](https://latex.codecogs.com/png.latex?\vec{x}_i), named nearest(![Formula](https://latex.codecogs.com/png.latex?\vec{x}_i)). 

For each bird $i$ with position ![Formula](https://latex.codecogs.com/png.latex?\vec{x}_i) and velocity ![Formula](https://latex.codecogs.com/png.latex?\vec{v}_i) four forces are computed:

- ![Formula](https://latex.codecogs.com/png.latex?\vec{f}_{separation}=-\lambda_{separation}\cdot(nearest(\vec{x}_i)-\vec{x}_i))
- ![Formula](https://latex.codecogs.com/png.latex?\vec{f}_{alignment}=\lambda_{alignment}\cdot(\vec{v}-\vec{v}_i))
- ![Formula](https://latex.codecogs.com/png.latex?\vec{f}_{coherence}=\lambda_{coherence}\cdot(\vec{x}-\vec{x}_i))
- ![Formula](https://latex.codecogs.com/png.latex?\vec{f}_{avoidance}=\lambda_{avoidance}\cdot\overrightarrow{center})

Where ![Formula](https://latex.codecogs.com/png.latex?\overrightarrow{center}) is the vector connecting each bird to the center of the simulation space.

The total force ![Formula](https://latex.codecogs.com/png.latex?\vec{f}_{total}=\vec{f}_{separation}+\vec{f}_{alignment}+\vec{f}_{coherence}+\vec{f}_{avoidance}) is used for updating birds position abd velocity:

-  ![Formula](https://latex.codecogs.com/png.latex?\vec{x}_i=\vec{x}_i+\vec{v}_i\cdot\Delta{t}+\frac{1}{2}\vec{f}_{total}\cdot\Delta{t^2})
-  ![Formula](https://latex.codecogs.com/png.latex?\vec{v}_i=\vec{v}_i+\vec{f}_{total}\cdot\Delta{t})
