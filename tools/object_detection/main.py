import hydra
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.datasets import DetectionDataset
from src.models import FasterRCNN
from src.trainers import DetectionTrainer


@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg):
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
    train_dataset = DetectionDataset(train=True, data_path=cfg.data_path, labels=cfg.labels,
                                     transforms=transforms_train)
    valid_dataset = DetectionDataset(train=False, data_path=cfg.data_path, labels=cfg.labels,
                                     transforms=transforms_val)

    # create the dataloaders
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                           collate_fn=train_dataset.collate_fn)
    val_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True,
                                         collate_fn=valid_dataset.collate_fn)

    # create the model
    model = FasterRCNN(n_classes=cfg.n_classes)

    # instantiate the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # initialize trainer
    trainer = DetectionTrainer(
        config=cfg,
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        optimizer=optimizer,
        loss_computed_by_model=cfg.loss_computed_by_model,
        n_classes=cfg.n_classes,
    )

    # start training
    best_metric = trainer.fit()
    trainer.evaluate()

    return best_metric


if __name__ == "__main__":
    main()
