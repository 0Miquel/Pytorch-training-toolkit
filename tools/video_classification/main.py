import hydra
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.datasets import UCFDataset
from src.models import CNNLSTM, Resnet50, ClassConvLSTM, ResNet3D
from src.trainers import ClassificationTrainer


@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg):
    # transforms
    transforms_train = A.Compose([
        A.Resize(width=224, height=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    transforms_val = A.Compose([
        A.Resize(width=224, height=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # create the model
    if cfg.model_name == "cnnlstm":
        model = CNNLSTM(n_classes=cfg.n_classes, backbone='resnet50')
    elif cfg.model_name == "resnet50":
        model = Resnet50(n_classes=cfg.n_classes)
        cfg.n_frames = 1
    elif cfg.model_name == "convlstm":
        model = ClassConvLSTM(n_classes=cfg.n_classes)
    elif cfg.model_name == "resnet3d":
        model = ResNet3D(n_classes=cfg.n_classes)

    # create the dataset
    train_dataset = UCFDataset(train=True, data_path=cfg.data_path, transforms=transforms_train, n_frames=cfg.n_frames)
    valid_dataset = UCFDataset(train=False, data_path=cfg.data_path, transforms=transforms_val, n_frames=cfg.n_frames)

    # create the dataloaders
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True)

    # create the loss function
    criterion = nn.CrossEntropyLoss()

    # instantiate the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.max_lr,
                                                    total_steps=cfg.n_epochs * len(train_dl))

    # initialize trainer
    trainer = ClassificationTrainer(
        config=cfg,
        train_dl=train_dl,
        val_dl=val_dl,
        criterion=criterion,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # start training
    best_metric = trainer.fit()

    return best_metric


if __name__ == "__main__":
    main()
