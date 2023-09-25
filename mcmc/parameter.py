from dataclasses import dataclass, field
from typing import Callable

import numpy as np

@dataclass
class Parameter():
    """
    """
    name: str
    initial_values: np.ndarray
    positive: bool = False
    bounded_below: float = False
    bounded_above: float = False
    transform_forwards: Callable[[float], float] = lambda x: x
    transform_backwards: Callable[[float], float] = lambda x: x
    values: np.ndarray = field(init=False)
    len: int = field(init=False, repr=False)
    prior_densities: np.ndarray = field(init=False)


    def __post_init__(self) -> None:
        if self.initial_values.ndim != 1:
            raise ValueError(f"initial_values must be 1-dimensional. {self.initial_values.ndim}-dimensional prodived")
        self.values = self.initial_values.astype('float32')
        self.len = len(self.initial_values)
        self.prior_densities = np.zeros(self.len)


    def __len__(self) -> int:
        return self.len


    def update(
        self, 
        index: int,
        new_value: float,
    ) -> None:
        """
        """
        self.values[index] = new_value


    def get_prior_density_sum(self) -> float:
        """
        """
        return np.sum(self.prior_densities)


    def __iter__(self):
        self._index = -1
        return self

    
    def __next__(self) -> float:
        self._index += 1
        if self._index == self.len:
            raise StopIteration
        return self.values[self._index]


    def is_proposal_acceptable(self, proposal: float) -> bool:
        """
        """
        transformed_proposal = self.transform_forwards(proposal)

        if self.positive == True and transformed_proposal <= 0:
            return False
        
        if self.bounded_below is not False and transformed_proposal < self.bounded_below:
            return False
        
        if self.bounded_above is not False and transformed_proposal > self.bounded_above:
            return False

        return True