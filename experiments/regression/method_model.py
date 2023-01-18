import os
import pathlib
import yaml

from data.dataset import Dataset
from experiments.experiment import Experiment

from tradeoffs.enums import TradeoffEnum
from models.enums import ModelEnum, MethodEnum

class MethodModel(Experiment):
    def __init__(self, yaml_file) -> None:
        self.path = ''
        config_path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'configs', yaml_file)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file) 

    def save_results(self):
        pass

    def run(self):
        
        # create dataset
        dataset_config = self.config['dimensions']['dataset']
        dataset = Dataset(**dataset_config)

        pass

MethodModel(yaml_file='experiment1.yaml')