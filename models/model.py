from abc import ABC, abstractclassmethod

class Model(ABC):

    @abstractclassmethod
    def fit(self, X, w, Y):
        pass

    @abstractclassmethod
    def predict(self, X):
        pass

    def fit_predict(self, X, w, Y):
        self.fit(X, w, Y)
        return self.predict(X)