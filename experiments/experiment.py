from abc import ABC, abstractclassmethod

class Experiment(ABC):
    pass

from models.uplift_modeling.direct_approach import CausalTree
from models.regression.problem_transformation.single_target import SingleTarget
from models.regression.problem_transformation.multi_target import MultiTarget
from models.regression.problem_transformation.regressor_chains import RegressorChain
from data.sampler import Sampler

sampler = Sampler()
data = sampler.sample()
print(f"data info: {data['X'].shape, data['W'].shape, data['Y_obs'].shape}")

# st_model = SingleTarget(base_estimator=CausalTree, base_estimator_kwargs={})
# st_model.fit(X=data["X"], w=data["W"], Y=data["Y_obs"])
# print(st_model.predict(X=data["X"]).shape)

# mt_model = MultiTarget(base_estimator=CausalTree, base_estimator_kwargs={})
# mt_model.fit(X=data["X"], w=data["W"], Y=data["Y_obs"])
# print(mt_model.predict(X=data["X"]).shape)

rc_model = RegressorChain(base_estimator=CausalTree, base_estimator_kwargs={})
rc_model.fit(X=data["X"], w=data["W"], Y=data["Y_obs"])
print(rc_model.predict(X=data["X"]).shape)