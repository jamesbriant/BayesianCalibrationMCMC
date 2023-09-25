from typing import Type
from copy import deepcopy

import numpy as np

from mcmc.data import Data
from mcmc.models.base import BaseModel
from mcmc.utilities import convert_to_nparray
# from mcmc.parameter import Parameter
from mcmc.chain import Chain
from mcmc.proposal import Proposal


class MCMC:
    """
    """
    def __init__(
        self, 
        max_iter: int, 
        model: Type[BaseModel], 
        data: Data, 
        proposal_widths: dict,
        **kwargs
    ):
        """
        """
        self._max_iter = max_iter
        self.model0 = model
        self._data = data
        self._proposal_widths = {}

        # self._proposal_widths = proposal_widths

        self.__dict__.update(kwargs)

        #Pass proposal widths to the parameters within the model.
        for name, values in proposal_widths.items():
            if name not in self.model0.get_param_names():
                raise ValueError(f"{name} is not a valid parameter name.")
            
            self._proposal_widths[name] = convert_to_nparray(values)

        self.model0.prepare_for_mcmc(self._data)


    def run(self) -> None:
        """
        """
        #for iteration
            #for each param
                #propose new param val
                #create model1 from model0
                #set proposal in model1
                #evaluate as necessary (m_d/V_d)
                #accept/reject new model

        self.chain = Chain(
            self._max_iter, 
            self.model0.get_param_long_names()
        )
        rng = np.random.default_rng()

        # #chain iterator
        # for iter in range(self._max_iter):
        #     if iter % 50 == 0:
        #         print(f"iteration: {iter}")
        #     #iterate over each parameter
        #     for model_param_name in self.model0.get_param_names():
        #         #iterate over each parameter's value
        #         param = self.model0.params[model_param_name]
        #         for index, param_value in enumerate(param):
        #             #Find a better way than deepcopy(). deepcopy() is slow. 
        #             #IDEA: set the proposal and have Model calculate the required parts under the proposal.

        #             #Making proposals should probably be moved back to the MCMC scheme. 
        #             #Combine this with the deepcopy() changes above.
        #             # proposal = model1_param.make_proposal(index, rng)
        #             proposal = param_value + (rng.random(1) - 0.5)*self._proposal_widths[model_param_name][index]
        #             if param.is_proposal_acceptable(proposal):                    
        #                 model1 = deepcopy(self.model0)
        #                 model1_param = model1.params[model_param_name]

        #                 model1.update(model1_param, index, proposal, self._data)

        #                 a = model1.logpost
        #                 b = self.model0.logpost
        #                 if iter < 2:
        #                     print(model_param_name, a, b)

        #                 if np.log(rng.random(1)) < a - b:
        #                     del self.model0
        #                     self.model0 = model1

        #     # self._samples[iter, :] = list(self.model0.get_model_params().values())
        #     self.chain.update(self.model0.get_model_params())







        #chain iterator
        for iter in range(self._max_iter):
            if iter % 50 == 0:
                print(f"Iteration: {iter}. Log-Posterior: {self.model0.logpost}")
                print(self.model0.get_model_params())
            #iterate over each parameter
            for model_param_name in self.model0.get_param_names():
                #iterate over each parameter's value
                param = self.model0.params[model_param_name]
                proposal = Proposal(param, self._proposal_widths[model_param_name], rng)
                for index in range(len(param)):
                # for index, param_value in enumerate(param):
                    #Find a better way than deepcopy(). deepcopy() is slow. 
                    #IDEA: set the proposal and have Model calculate the required parts under the proposal.

                    #Making proposals should probably be moved back to the MCMC scheme. 
                    #Combine this with the deepcopy() changes above.
                    # proposal = model1_param.make_proposal(index, rng)
                    # proposal = param_value + (rng.random(1) - 0.5)*self._proposal_widths[model_param_name][index]
                    # proposal = param.values[index] + (rng.random(1) - 0.5)*self._proposal_widths[model_param_name][index]
                    new_value = proposal.make_proposal(index)
                    if param.is_proposal_acceptable(new_value):
                        model1 = deepcopy(self.model0)
                        model1_param = model1.params[model_param_name]
                        model1.update(model1_param, index, new_value, self._data)
                        a = model1.logpost
                        b = self.model0.logpost

                        # if model1.loglike > self.model0.loglike:
                        #     print(iter, model_param_name, self.model0.loglike, model1.loglike)
                        #     pass

                        if np.log(rng.random(1)) < a - b:
                            del self.model0
                            self.model0 = model1

            self.chain.update(self.model0.get_model_params())