import hydra
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.datasets import FolderDataset
from src.models import Resnet
from src.trainers import ClassificationTrainer


@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg):
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
    train_dataset = FolderDataset(train=True, data_path=cfg.data_path, labels=cfg.labels,
                                  transforms=transforms_train)
    valid_dataset = FolderDataset(train=False, data_path=cfg.data_path, labels=cfg.labels,
                                  transforms=transforms_val)

    # create the dataloaders
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True)

    # create the model
    model = Resnet(n_classes=cfg.n_classes)

    # create the loss function
    criterion = nn.CrossEntropyLoss()

    # instantiate the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.max_lr,
    #                                                 total_steps=cfg.n_epochs * len(train_dl))

    # initialize trainer
    trainer = ClassificationTrainer(
        config=cfg,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=val_dl,
        criterion=criterion,
        model=model,
        optimizer=optimizer,
        # scheduler=scheduler,
    )

    # start training
    best_metric = trainer.fit()

    return best_metric


if __name__ == "__main__":
    main()
