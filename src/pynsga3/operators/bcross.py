from typing import Sequence
from abc import ABC, abstractmethod

import numpy as np

__all__ = ["CrossoverOp"]


class CrossoverOp(ABC):
    @abstractmethod
    def cross(self, parents: Sequence[np.ndarray], **kwargs):
        pass
