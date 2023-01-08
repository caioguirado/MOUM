from abc import ABC, abstractclassmethod

class Experiment(ABC):
    pass

from models.uplift_modeling.direct_approach import CausalTree
from models.regression.problem_transformation.single_target import SingleTarget
from data.sampler import Sampler

sampler = Sampler()
data = sampler.sample()
st_model = SingleTarget(base_estimator=CausalTree, base_estimator_kwargs={})

print(f"data info: {data['X'].shape, data['W'].shape, data['Y_obs'].shape}")
st_model.fit(X=data["X"], w=data["W"], Y=data["Y_obs"])
print(st_model.predict(X=data["X"]).shape)