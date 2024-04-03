# Pytorch training toolkit

This toolkit integrates the most well-known frameworks for machine learning like WandB for experiment 
tracking, Hydra for configuration management or Optuna for hyperparameter search.

In addition to that, the code is meant to be fully extensible, it should be possible to
implement your own custom models/losses/optimizers without changing 
existing code.

## Overview
- [Frameworks](#frameworks)
- [Installation](#installation)
- [Train](#train)

## Frameworks
Brief introduction to the frameworks that are integrated into the pipeline.

### WandB
WandB is an experiment tracking tool for machine learning.

Using WandB in the pipeline you will be able to:
- Track the hyperparameters and metrics of every run
- Display the predictions through the different epochs
- Upload complex media and charts
- Save your model as a WandB artifact

For more information on how it works, visit its [documentation](https://docs.wandb.ai/).


### Hydra
Hydra is a framework that simplifies the development of research and other complex applications.

Using Hydra in the pipeline you will be able to have a modular configuration schema in order 
to build your experiment configurations easier.

The Hydra configuration schema is the following:
```
conf
├── config.yaml             # config.yaml calls one .yaml file for 
├── dataset                 # every module inside the schema
│   ├── MNIST.yaml          # the .yaml files inside every module
│   └── cifar10.yaml        # specifies the configuration of that module   
├── model                   
│   ├── resnet.yaml         
│   └── ...   
└── ...
```
For more information on how it works, visit its [documentation](https://hydra.cc/docs/intro/).


### Optuna
In order to use Optuna together with Hydra we first need to install the Optuna Sweeper plugin.
```commandline
pip install hydra-optuna-sweeper --upgrade
```
It is needed to set the optuna sweeper in the config file, but it is already done by default in `config.yaml`.
```yaml
defaults:
  - override hydra/sweeper: optuna
```
Further changes can be added in the config file like the number of runs to execute or
whether we want to maximize or minimize our objective metric.

To run a sweeper we need to use a command like this one, here we will be executing multiple 
runs with different learning rates.
```commandline
python train.py --multirun 'optimizer.settings.lr=choice(0.1, 0.01, 0.001, 0.0001)'
```
For more information on how to create sweepers visit its [documentation](https://hydra.cc/docs/plugins/optuna_sweeper/).

## Train
CLI command to run a training experiment.
```
cd tools
python train.py 
```
Additional changes in the Hydra configuration can be added in the command line 
in the following way:
```
python train.py optimizer.settings.lr=0.1 trainer.wandb=test_project
```
In this case we have changed the default learning rate in the optimizer to 0.1 and
have set a wandb project name in order to log the results from the experiment,
if no project name is given it will not log any results into wandb.
