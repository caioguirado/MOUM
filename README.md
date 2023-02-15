# MOUM
Multi-Output Uplift Modeling

This package implements the structure necessary to run Multi-Output Uplift Modeling experiments.

To create a new experiment, clone one of the templates from `./experiments/configs/` and adapt it to your use case.

To run an experiment, execute the command:

`python3 -m MOUM.run_experiment <experiment_name>`

where `<experiment_name>` is the name of the `.yaml` file in `./experiments/configs/`. The experiment results will be saved under the `./experiment/results/` path.

To create your own data, evaluation, model or trade off type, implement them under the respective folders, and add them to the `./enums.py` file under each subpackage.