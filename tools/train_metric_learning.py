import hydra
import torch
from hydra.core.config_store import ConfigStore
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner

from src.config import Configuration
from src.datasets import FolderDataset
from src.models import Resnet18
from src.trainers import MetricLearningTrainer

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config_metric_learning", node=Configuration)


@hydra.main(version_base=None, config_path="configs", config_name="config_metric_learning.yaml")
def main(config: Configuration) -> None:
    # transforms
    transforms_train = A.Compose([
        A.Resize(width=64, height=64),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    transforms_val = A.Compose([
        A.Resize(width=64, height=64),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # create the dataset
    train_dataset = FolderDataset(train=True, data_path=config.data_path, labels=config.labels,
                                  transforms=transforms_train)
    valid_dataset = FolderDataset(train=False, data_path=config.data_path, labels=config.labels,
                                  transforms=transforms_val)

    # create the dataloaders
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)

    # create the model
    model = Resnet18(pretrained=config.pretrained, fine_tune=config.fine_tune, n_classes=config.n_classes)

    # create the loss function
    criterion = TripletMarginLoss()

    # create the miner function
    miner = TripletMarginMiner()

    # instantiate the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.max_lr,
                                                    total_steps=config.n_epochs * len(train_dl))

    # initialize trainer
    trainer = MetricLearningTrainer(
        config=config,
        train_dl=train_dl,
        val_dl=val_dl,
        criterion=criterion,
        model=model,
        optimizer=optimizer,
        miner=miner,
        scheduler=scheduler
    )

    # start training
    best_metric = trainer.fit()

    return best_metric


if __name__ == "__main__":
    main()
