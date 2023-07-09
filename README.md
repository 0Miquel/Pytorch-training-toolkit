# ToFu
<img src="media/tofu.png" alt="drawing" width="400"/>

TorchFusion (ToFu) is a modular pytorch training pipeline.

It integrates the most well-known frameworks for machine learning like WandB for experiment 
tracking, Hydra for configuration management or Optuna for hyperparameter search.

In addition to that, the code is meant to be fully extensible, it should be possible to
implement your own custom models/losses/optimizers without changing 
existing code.


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
│   └── cifar10.yaml        # specify the configuration of that module   
├── model                   
│   ├── resnet.yaml         
│   └── ...   
└── ...
```
For more information on how it works, visit its [documentation](https://hydra.cc/docs/intro/).


### Optuna
TO-DO


# Installation
Follow the next command lines to ensure you have ToFu installed.
```commandline
git clone https://github.com/0Miquel/ToFu.git
cd ToFu
pip install -r requirements.txt
pip install .
```

# Train 
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
