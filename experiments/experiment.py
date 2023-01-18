from abc import ABC, abstractclassmethod

class Experiment(ABC):

    @abstractclassmethod
    def save_results(self):
        pass

    @abstractclassmethod
    def run(self):
        # experiment
            # experiment type (mainly 2 - regression and rank)
            # load parameters
            # train models
            # evaluate
            # save/cache results
        pass