from tqdm import tqdm
import hydra
import os
import time
import torchvision.utils as vutils
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from src.utils import (
    load_batch_to_device,
    weights_init,
    plot_fake_imgs,
    Logger,
    save_model,
    MetricMonitor,
    set_random_seed
)
from src.datasets import get_dataloaders
from src.losses import get_loss
from src.models import get_model
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler

import torch


class DCGANTrainer:
    def __init__(self, config):
        set_random_seed(42)

        self.config = config
        trainer_config = config["trainer"]
        self.n_epochs = trainer_config["n_epochs"]
        self.save_media_epoch = self.n_epochs // 10 if self.n_epochs // 10 > 0 else 1
        self.device = trainer_config["device"]
        self.latent_vector_size = config['model']['generator']["settings"]["latent_vector"]

        self.logger = Logger(config)

        # DATASET
        dataloaders = get_dataloaders(config['dataset'], config["transforms"])
        self.train_dl = dataloaders["train"]

        # LOSS
        self.loss = get_loss(config['loss'])

        # MODEL
        self.netG = get_model(config['model']['generator']).to(self.device)
        # self.netG = torch.compile(self.netG)
        self.netG.apply(weights_init)
        self.netD = get_model(config['model']['discriminator']).to(self.device)
        # self.netD = torch.compile(self.netD)
        self.netD.apply(weights_init)

        # OPTIMIZER
        total_steps = len(self.train_dl) * self.n_epochs
        self.optimizerG = get_optimizer(config['optimizer'], self.netG)
        self.optimizerD = get_optimizer(config['optimizer'], self.netD)
        self.schedulerG = get_scheduler(config['scheduler'], self.optimizerG, total_steps) \
            if "scheduler" in config.keys() else None
        self.schedulerD = get_scheduler(config['scheduler'], self.optimizerD, total_steps) \
            if "scheduler" in config.keys() else None

    def train_epoch(self, epoch):
        self.netG.train()
        self.netD.train()
        metric_monitor = MetricMonitor()
        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.
        # use tqdm to track progress
        with tqdm(self.train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} train")
            # Iterate over data.
            for step, batch in enumerate(tepoch):
                batch = load_batch_to_device(batch, self.device)
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_cpu = batch['x']
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.loss(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.latent_vector_size, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.loss(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()
                # Update scheduler
                if self.schedulerD is not None:
                    self.schedulerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.loss(output, label)
                # Calculate gradients for G
                errG.backward()
                # Update G
                self.optimizerG.step()
                # Update scheduler
                if self.schedulerG is not None:
                    self.schedulerG.step()

                # compute epoch metrics and loss
                metric_monitor.update("lossD", errD.item())
                metric_monitor.update("lossG", errG.item())
                metrics = metric_monitor.get_metrics()
                metrics["lrG"] = self.optimizerG.param_groups[0]['lr']
                metrics["lrD"] = self.optimizerD.param_groups[0]['lr']
                tepoch.set_postfix(metrics)

        if epoch % self.save_media_epoch == 0:
            fake_imgs = plot_fake_imgs(self.netG, self.latent_vector_size)
            self.logger.add_media({"fake_imgs": fake_imgs})
        self.logger.add_metrics(metrics, "train")

    def fit(self):
        since = time.time()

        for epoch in range(self.n_epochs):
            self.train_epoch(epoch)
            save_model(self.netG, 'last_epoch_generator.pt')
            save_model(self.netD, 'last_epoch_discriminator.pt')
            if self.logger is not None:
                self.logger.upload(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        if self.logger is not None:
            self.logger.finish()
