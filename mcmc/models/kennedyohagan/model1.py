from mcmc.data import Data
from mcmc.models.kennedyohagan.kennedyohagan import KennedyOHagan
from mcmc.parameter import Parameter
from mcmc.utilities import dist_matrix3 as dist_matrix
# from mcmc.utilities import obs_sim_dist
from mcmc.utilities import obs_sim_dist_jitted as obs_sim_dist

import numpy as np
from scipy.stats import beta, gamma

import jax.numpy as jnp
from jax import jit

class Model(KennedyOHagan):
    """
    """
    def __init__(
        self, 
        params: dict,
        *args, 
        **kwargs
    ):
        """
        """
        #DO NOT DELETE THIS LINE
        super().__init__(params, *args, **kwargs)


    def prepare_for_mcmc(self, data: Data) -> None:
        """Evaluates the model's log-posterior given the initial parameter values.
        Ensure all intermediary steps for calculating the model posterior
        are placed in this method.
        """
        # Place ALL of your custom methods here which are used to calculate m_d and V_D.
        # They must appear in the correct order.

        ############################################################
        #Place your code here
        
        self.D_eta, self.D_delta, self.D_B_I = dist_matrix(data, self.params['theta'])
        self.I_eta = np.triu_indices(n=data.n+data.m, k=1)
        self.I_delta = np.triu_indices(n=data.n, k=1)

        ############################################################

        #The next line calculates and saves the prior densities.
        #DO NOT DELETE THIS LINE
        super().prepare_for_mcmc(data)

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
        #DO NOT DELETE THIS LINE
        super().update(param, index, new_value, data)

        ############################################################
        #Place your code here

        if param.name == "theta":
            # self.D_eta[self.D_B_I, data.p+index] = obs_sim_dist(data, param.values[index]).flatten()

            self.D_eta[self.D_B_I, data.p+index] = obs_sim_dist(
                data.t, 
                jnp.tile(param.values[index], data.n).reshape(-1,1)
            ).block_until_ready()

            self.calc_m_d(data)
            self.calc_sigma_eta(data)
            self.calc_V_d(data)
        elif param.name == "omega_eta":
            self.calc_sigma_eta(data)
            self.calc_V_d(data)
        elif param.name == "omega_delta":
            self.calc_sigma_delta(data)
            self.calc_V_d(data)
        elif param.name == "lambda_eta":
            self.calc_sigma_eta(data)
            self.calc_V_d(data)
        elif param.name == "lambda_delta":
            self.calc_sigma_delta(data)
            self.calc_V_d(data)
        elif param.name == "lambda_epsilon":
            self.calc_sigma_epsilon(data)
            self.calc_V_d(data)
        elif param.name == "lambda_epsilon_eta":
            self.calc_sigma_epsilon_eta(data)
            self.calc_V_d(data)

        ############################################################

        #DO NOT DELETE THIS LINE
        self.calc_logpost(data)


    def calc_prior(self, param: Parameter) -> None:
        """Evaluates the prior for the given parameter using the object's attribute value."""
        ############################################################
        #Place your code here

        if param.name == "theta":
            pass
        elif param.name == "omega_eta":
            omega_eta = param.values.copy()
            omega_eta[omega_eta > 0.999] = 0.999    #copy() prevents overwriting the parameter values!
            for index, value in enumerate(omega_eta):
                param.prior_densities[index] = beta.logpdf(value, 1, 0.5, loc=0, scale=1)
        elif param.name == "omega_delta":
            omega_delta = param.values.copy()       #copy() prevents overwriting the parameter values!
            omega_delta[omega_delta > 0.999] = 0.999
            for index, value in enumerate(omega_delta):
                param.prior_densities[index] = beta.logpdf(value, 1, 0.4, loc=0, scale=1)
        elif param.name == "lambda_eta":
            for index, value in enumerate(param):
                param.prior_densities[index] = gamma.logpdf(value, a=10, scale=1/10)
        elif param.name == "lambda_delta":
            for index, value in enumerate(param):
                param.prior_densities[index] = gamma.logpdf(value, a=10, scale=1/0.3)
        elif param.name == "lambda_epsilon":
            for index, value in enumerate(param):
                param.prior_densities[index] = gamma.logpdf(value, a=10, scale=1/0.001)
        elif param.name == "lambda_epsilon_eta":
            for index, value in enumerate(param):
                param.prior_densities[index] = gamma.logpdf(value, a=10, scale=1/0.001)

        ############################################################

        #DO NOT DELETE THIS LINE
        super().calc_prior(param)

    
    def calc_m_d(self, data: Data) -> None:
        """
        """
        ############################################################
        #Place your code here

        self.m_d = np.zeros((data.n + data.m, 1))

        ############################################################
        #DO NOT DELETE THIS LINE
        # super().calc_m_d(data)


    def calc_sigma_eta(self, data: Data) -> None:
        """
        """
        # self._sigma_eta = np.zeros((data.n+data.m, data.n+data.m))
        # beta_eta = -4*np.log(self.params['omega_eta'].values)
        # self._sigma_eta[self.I_eta] += np.exp(-self.D_eta @ beta_eta)/self.params['lambda_eta'].values[0]
        # self._sigma_eta += self._sigma_eta.T
        # self._sigma_eta += (1/self.params['lambda_eta'].values[0]) * np.eye(data.n+data.m)

        self._sigma_eta = np.array(calc_sigma_eta(
            jnp.zeros((data.n+data.m, data.n+data.m)),
            jnp.eye(data.n+data.m), 
            self.params['omega_eta'].values,
            self.params['lambda_eta'].values[0],
            self.D_eta,
            self.I_eta
        ))
    
    def calc_sigma_delta(self, data: Data) -> None:
        """
        """
        # self._sigma_delta = np.zeros((data.n, data.n))
        # beta_delta = -4*np.log(self.params['omega_delta'].values)
        # self._sigma_delta[self.I_delta] = np.exp(-self.D_delta @ beta_delta)/self.params['lambda_delta'].values
        # self._sigma_delta += self._sigma_delta.T
        # self._sigma_delta += (1/self.params['lambda_delta'].values) * np.eye(data.n)

        self._sigma_delta = np.eye(data.n)/self.params['lambda_delta'].values[0]


    def calc_sigma_epsilon(self, data: Data) -> None:
        """
        """
        self._sigma_epsilon = np.eye(data.n)/self.params['lambda_epsilon'].values[0]

    
    def calc_sigma_epsilon_eta(self, data: Data) -> None:
        """
        """
        self._sigma_epsilon_eta = np.eye(data.m)/self.params['lambda_epsilon_eta'].values[0]


    def calc_logpost(self, data: Data):
        b = self.params['lambda_epsilon_eta'].values[0]
        if b>2e5 or b<100.:
            self.logpost = -9e99
        else:
            super().calc_logpost(data=data)


# @jit
# def calc_sigma_eta(nm, omega_eta, lambda_eta, D_eta, I_eta) -> None:
#     """
#     """
#     beta_eta = -4*jnp.log(omega_eta)

#     _sigma_eta = jnp.zeros((nm, nm))
#     _sigma_eta = _sigma_eta.at[I_eta].add(jnp.exp(jnp.matmul(-D_eta, beta_eta))/lambda_eta)
#     _sigma_eta = _sigma_eta + _sigma_eta.T
#     _sigma_eta = _sigma_eta + (1/lambda_eta)*jnp.eye(nm)

#     return _sigma_eta

@jit
def calc_sigma_eta(sigma_eta, eye, omega_eta, lambda_eta, D_eta, I_eta) -> None:
    """
    """
    beta_eta = -4*jnp.log(omega_eta)

    _sigma_eta = sigma_eta.at[I_eta].add(jnp.exp(jnp.matmul(-D_eta, beta_eta))/lambda_eta)
    _sigma_eta = _sigma_eta + _sigma_eta.T
    _sigma_eta = _sigma_eta + (1/lambda_eta)*eye

    return _sigma_eta