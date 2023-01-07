from abc import ABC, abstractclassmethod

class Model(ABC):

    @abstractclassmethod
    def fit():
        pass

    @abstractclassmethod
    def predict():
        pass