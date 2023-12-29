# GP-toy-problems

Python implementation of Matlab code for the Kennedy & O'Hagan (2001) framework for calibration of computer simulators.

A generic Metropolis-Hastings random walk MCMC tool for testing a python implementation of a simple Kennedy & O'Hagan model written in MATLAB.

## Custom Models

Custom models can be implementated by inheriting from `BaseModel()` in `mcmc/models/base.py` and `KennedyOHagan()` in `mcmc/models/kennedyohagan/kennedyohagan.py`.

## Examples

`simple1.py` provides an example of how this tool works. See also `simple1.ipynb` for an interactive notebook of this same example.