import numpy as np
from abc import ABC, abstractmethod

class DecisionMapping(ABC):
    def check_input(self, probabilities):
        assert type(probabilities) is np.ndarray
        assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all()

    @abstractmethod
    def __call__(self, probabilities):
        pass

class ImageNetProbabilitiesTo1000ClassesMapping(DecisionMapping):
    """Return the WNIDs sorted by probabilities."""
    def __init__(self):
        self._get_ilsvrc2012_WNIDs()
    
    def _get_ilsvrc2012_WNIDs(self):
        with open('./decision_mapping/imagenet_categories.txt') as f:
            self.categories = [line.split()[0] for line in f]

    def __call__(self, probabilities):
        self.check_input(probabilities)
        sorted_indices = np.flip(np.argsort(probabilities), axis=-1)
        # print(probabilities)=
        return np.take(self.categories, sorted_indices, axis=-1)
