# Bayesian Calibration MCMC

A generic Metropolis-Hastings random walk MCMC tool for testing a python implementation of the Kennedy & O'Hagan (2001)[^1] Bayesian Calibration for Computer Models framework. The implementation design was inspired by a MATLAB implementation from the early 2000s.

## Custom Models

Custom models can be implementated by inheriting from `BaseModel()` in `mcmc/models/base.py` and `KennedyOHagan()` in `mcmc/models/kennedyohagan/kennedyohagan.py`.

## Examples

`simple1.py` provides an example of how this tool works. See also `simple1.ipynb` for an interactive notebook of this same example.

## Implementation Details

This approach seeks to save as many components as possible between each MCMC iteration to avoid calculating the same components more than once.
This implementation relies only on numpy as scipy. A version which uses JAX is in partial development, but is unlikely to see further development. Instead, visit [jamesbriant/koh-gpjax](https://github.com/jamesbriant/KOH-GPJax).

## References
[^1]: Kennedy, M.C. and O'Hagan, A. (2001), Bayesian calibration of computer models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63: 425-464. https://doi.org/10.1111/1467-9868.00294