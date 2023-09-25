from typing import Dict, List
from abc import ABC, abstractmethod

import numpy as np

from mcmc.parameter import Parameter
from mcmc.data import Data


class BaseModel(ABC):
    """Class representing a the minimum requirements for the MCMC operations.
    Build on top of this class.
    """

    #ORDERED LIST FOR MCMC
    _accepted_params = []


    @abstractmethod
    def __init__(
        self, 
        params: Dict[str, Parameter], 
        *args, 
        **kwargs
    ):
        """
        """

        if set(params.keys()) != set(self._accepted_params):
            raise ValueError(f"Parameter names must match exactly for the chosen model. {self._accepted_params}")

        self.params = params
        self._prior_densities = {}

        self.total_param_count_long = 0
        for param in self.params.values():
            self.total_param_count_long += len(param)

        self.__dict__.update(kwargs)


    @abstractmethod
    def prepare_for_mcmc(self, data: Data) -> None:
        """Evaluates the model's log-posterior given the initial parameter values.
        Subclasses should ensure all intermediary steps for calculating the model posterior
        are placed in this method.
        """
        #Calculate priors
        for param in self.params.values():
            self.calc_prior(param)
            
        self.calc_logpost(data)


    @abstractmethod
    def update(
        self, 
        param: Parameter, 
        index: int,
        new_value: float,
        data: Data,
    ) -> None:
        """Updates the Parameter value and recalculates the prior distribution.
        Subclasses should also update any additional requirements, such as the log-posterior.
        """
        param.update(index, new_value)
        self.calc_prior(param)


    @abstractmethod
    def calc_loglike(self, data: Data) -> None:
        """This method depends on your model!
        """
        pass


    @abstractmethod
    def calc_prior(self, param: Parameter):
        """Evaluates the prior for the given parameter using the object's attribute value."""
        self._prior_densities[param.name] = param.get_prior_density_sum()

    
    def calc_logpost(self, data: Data):
        """Evaluates the log-posterior of the model given the current parameter values."""
        self.calc_loglike(data)
        self.logpost = np.sum(list(self._prior_densities.values())) + self.loglike


    def get_param_names(self) -> List[str]:
        """Returns set of parameter names.
        """
        return self._accepted_params

    
    def get_param_long_names(self) -> List[str]:
        """Returns set of parameter names with index numbers appended.
        """
        output = []
        for name, param in self.params.items():
            for i in range(len(param)):
                output.append(f"{name}_{i}")
        print(output)
        return output


    def get_model_params(self) -> dict:
        """Returns dictionary of parameter names and values.
        """
        output = {}
        for name, param in self.params.items():
            output.update(
                zip(
                    [f"{name}_{i}" for i in range(len(param))],
                    param.values
                )
            )
        return output