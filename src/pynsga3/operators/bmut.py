from abc import ABC, abstractmethod

import numpy as np

__all__ = ["MutationOp"]


class MutationOp(ABC):
    @abstractmethod
    def mutate(self, individual: np.ndarray, **kwargs) -> bool:
        pass
