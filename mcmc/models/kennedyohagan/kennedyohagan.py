from abc import abstractmethod

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from mcmc.data import Data
from mcmc.models.base import BaseModel
from mcmc.parameter import Parameter

class KennedyOHagan(BaseModel):
    """Implementation of the Kennedy & O'Hagan (2001) model.
    """
    # ORDERED LIST FOR MCMC
    _accepted_params = [
        "theta",
        "omega_eta",
        "omega_delta",
        "lambda_eta",
        "lambda_epsilon_eta",
        "lambda_delta",
        "lambda_epsilon"
    ]


    def __init__(
        self, 
        params: dict,
        *args, 
        **kwargs
    ):
        """
        """
        super().__init__(params, *args, **kwargs)

        self.m_d = None
        self.V_d = None
        self.V_d_chol = None

        self._sigma_eta = None
        self._sigma_delta = None
        self._sigma_epsilon = None
        self._sigma_epsilon_eta = None

    
    def prepare_for_mcmc(self, data: Data) -> None:
        """Evaluates the model's log-posterior given the initial parameter values.
        Subclasses should ensure all intermediary steps for calculating the model posterior
        are placed in this method.
        """
        # Place ALL of your custom methods here which are used to calculate m_d and V_D.
        # They must appear in the correct order.

        self.calc_sigma_eta(data)
        self.calc_sigma_delta(data)
        self.calc_sigma_epsilon(data)
        self.calc_sigma_epsilon_eta(data)

        self.calc_m_d(data)
        self.calc_V_d(data)

        #The next line calculates and saves the prior densities.
        #DO NOT DELETE THIS LINE
        super().prepare_for_mcmc(data)


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
        super().update(param, index, new_value, data)
        # self.calc_logpost(data) # not needed as logpost calulated in model1.py


    @abstractmethod
    def calc_prior(self, param: Parameter) -> None:
        """Evaluates the prior for the given parameter using the object's attribute value."""

        ############################################################

        ############################################################

        #DO NOT DELETE THIS LINE
        super().calc_prior(param)
        

    def calc_m_d(self, data: Data) -> None:
        """
        """
        pass        


    def calc_V_d(self, data: Data) -> None:
        """
        """
        self.V_d = np.zeros((data.n+data.m, data.n+data.m))
        self.V_d += self._sigma_eta
        self.V_d[:data.n, :data.n] += self._sigma_delta + self._sigma_epsilon
        self.V_d[data.n:, data.n:] += self._sigma_epsilon_eta
        
        self.V_d_chol = cho_factor(self.V_d)

        # try:
        #     self.V_d_chol = np.linalg.cholesky(self.V_d)
        # except Exception as e:
        #     print("The eigenvalues of self.V_d are:")
        #     print(np.linalg.eigvalsh(self.V_d))
        #     raise e
        
    
    @abstractmethod
    def calc_sigma_eta(self, data: Data) -> None:
        """
        """
        pass


    @abstractmethod
    def calc_sigma_delta(self, data: Data) -> None:
        """
        """
        pass


    @abstractmethod
    def calc_sigma_epsilon(self, data: Data) -> None:
        """
        """
        pass


    @abstractmethod
    def calc_sigma_epsilon_eta(self, data: Data) -> None:
        """
        """
        pass


    def calc_loglike(self, data: Data) -> None:
        """Kennedy & O'Hagan model log-likelihood
        """
        # u = np.linalg.solve(self.V_d_chol, data.d - self.m_d)
        u = cho_solve(self.V_d_chol, data.d - self.m_d)
        Q = np.sum(u**2)
        # logdet = np.sum(np.log(np.diag(self.V_d_chol)))
        logdet = np.sum(np.log(np.diag(self.V_d_chol[0])))
        self.loglike = -logdet - 0.5*Q