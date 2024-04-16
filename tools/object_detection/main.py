import sys
sys.path.append('../../')

import hydra
import torch
import torch.nn as nn

from hydra.core.config_store import ConfigStore
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import Configuration
from src.datasets import DetectionDataset
from src.models import FasterRCNN
from src.trainers import DetectionTrainer

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Configuration)


@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(config: Configuration) -> None:
    # transforms
    transforms_train = A.Compose([
        A.Resize(width=224, height=224),
        # A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["labels"]))
    transforms_val = A.Compose([
        A.Resize(width=224, height=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["labels"]))

    # create the dataset
    train_dataset = DetectionDataset(train=True, data_path=config.data_path, labels=config.labels,
                                     transforms=transforms_train)
    valid_dataset = DetectionDataset(train=False, data_path=config.data_path, labels=config.labels,
                                     transforms=transforms_val)

    # create the dataloaders
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                           collate_fn=train_dataset.collate_fn)
    val_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True,
                                         collate_fn=valid_dataset.collate_fn)

    # create the model
    model = FasterRCNN(n_classes=config.n_classes, pretrained=config.pretrained)

    # instantiate the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.max_lr,
                                                    total_steps=config.n_epochs * len(train_dl))

    # initialize trainer
    trainer = DetectionTrainer(
        config=config,
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        optimizer=optimizer,
        loss_computed_by_model=config.loss_computed_by_model,
        n_classes=config.n_classes,
        scheduler=scheduler
    )

    # start training
    best_metric = trainer.fit()

    return best_metric


if __name__ == "__main__":
    main()
