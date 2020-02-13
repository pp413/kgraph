from abc import ABC, abstractmethod

class Module(ABC):
    @abstractmethod
    def score(self, *args):
        pass
    
    @abstractmethod
    def fit(self, *args):
        pass
    
    @abstractmethod
    def evaluate(self, *args):
        pass
