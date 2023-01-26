import torch.optim as optim


def get_optimizer(config, model):
    optimizer_name = config['optimizer_name']
    settings = config['settings']

    try:
        optimizer = getattr(optim, optimizer_name)(model.parameters(), **settings)
    except AttributeError:
        try:
            optimizer = globals()[optimizer_name](model.parameters(), **settings)
        except KeyError:
            raise f"Optimizer with name {optimizer_name} not found"

    return optimizer


class MyOptimizer(optim.Optimizer):
    def __init__(self, params, **kwargs):
        # Call the parent class's constructor
        super(MyOptimizer, self).__init__(params, **kwargs)
        # Set any additional hyperparameters you need
        self.learning_rate = kwargs.get('learning_rate', 0.01)

    def step(self):
        # Implement the optimization step for each parameter
        for param in self.param_groups:
            param.data -= self.learning_rate * param.grad
