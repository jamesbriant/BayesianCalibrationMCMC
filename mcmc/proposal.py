import numpy as np

from mcmc.parameter import Parameter

class Proposal:
    """
    """
    def __init__(
        self,
        param: Parameter,
        proposal_widths: np.ndarray,
        rng: np.random.Generator,
    ):
        """
        """
        self.param = param
        self.proposal_widths = proposal_widths
        self.rng = rng

    def make_proposal(self, index: int) -> float:
        """
        """
        # value = self.param.transform_forwards(self.param.values[index])
        # value += (self.rng.random(1) - 0.5)*self.proposal_widths[index]
        # return self.param.transform_backwards(value)
    
        # return self.param.values[index] + (self.rng.random(1) - 0.5)*self.proposal_widths[index]

        value = self.param.transform_forwards(self.param.values[index])
        diff = (self.rng.random(1) - 0.5)*self.proposal_widths[index]
        try:
            return self.param.transform_backwards(value + diff)
        except:
            return value