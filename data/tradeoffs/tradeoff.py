from abc import ABC, abstractclassmethod

class Tradeoff(ABC):

    @abstractclassmethod
    def __init__(self) -> None:
        pass 

    @abstractclassmethod
    def get_main_effect(self):
        pass

    @abstractclassmethod
    def get_tradeoff_effect(self):
        pass

    @abstractclassmethod
    def create_Y(self, X, n_responses):
        pass