from tqdm import tqdm
import hydra
import os
import time
import torchvision.utils as vutils
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from tofu.utils import load_batch_to_device, weights_init, save_fake_imgs, Logger
from tofu.datasets import get_dataloaders
from tofu.losses import get_loss
from tofu.models import get_model
from tofu.optimizers import get_optimizer
from tofu.schedulers import get_scheduler

import torch


class DCGANTrainer:
    def __init__(self, config):
        self.config = config
        trainer_config = config["trainer"]
        self.n_epochs = trainer_config["n_epochs"]
        self.device = trainer_config["device"]
        self.latent_vector_size = config['model']['generator']["settings"]["latent_vector"]

        self.logger = None
        if trainer_config["wandb"] is not None:
            self.logger = Logger(config)

        # DATASET
        dataloaders = get_dataloaders(config['dataset'], config["transforms"])
        self.train_dl = dataloaders["train"]

        # LOSS
        self.loss = get_loss(config['loss'])

        # MODEL
        netG = get_model(config['model']['generator']).to(self.device)
        self.netG = torch.compile(netG)
        self.netG.apply(weights_init)
        netD = get_model(config['model']['discriminator']).to(self.device)
        self.netD = torch.compile(netD)
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
        running_lossG = 0
        running_lossD = 0
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
                real_cpu = batch['imgs']
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
                # COMPUTE EPOCHS LOSS
                running_lossD += errD.item()
                epoch_lossD = running_lossD / (step + 1)
                # Update D
                self.optimizerD.step()

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
                # COMPUTE EPOCHS LOSS
                running_lossG += errG.item()
                epoch_lossG = running_lossG / (step + 1)
                # Update G
                self.optimizerG.step()

                # UPDATE SCHEDULERS
                if self.schedulerD is not None:
                    self.schedulerD.step()
                if self.schedulerG is not None:
                    self.schedulerG.step()
                # LEARNING RATE
                current_lrG = self.optimizerG.param_groups[0]['lr']
                current_lrD = self.optimizerD.param_groups[0]['lr']
                tepoch.set_postfix({"lossG": epoch_lossG, "lossD": epoch_lossD, "lrG": current_lrG, "lrD": current_lrD})

        fixed_noise = torch.randn(64, self.latent_vector_size, 1, 1, device=self.device)
        fake = self.netG(fixed_noise).detach().cpu()
        fake_imgs = vutils.make_grid(fake, padding=2, normalize=True)
        save_fake_imgs(fake_imgs, epoch)
        if self.logger is not None:
            self.logger.add({"lossG": epoch_lossG, "lossD": epoch_lossD, "lrG": current_lrG, "lrD": current_lrD}, "train")

    def fit(self):
        since = time.time()

        for epoch in range(self.n_epochs):
            self.train_epoch(epoch)
            self.save_model(self.netG, 'last_epoch_generator.pt')
            self.save_model(self.netD, 'last_epoch_discriminator.pt')
            if self.logger is not None:
                self.logger.upload()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        if self.logger is not None:
            self.logger.finish()

    @staticmethod
    def save_model(model, model_name):
        outputs_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        ckpts_dir = os.path.join(outputs_dir, 'ckpts')
        os.makedirs(ckpts_dir, exist_ok=True)
        model_path = os.path.join(ckpts_dir, model_name)
        torch.save(model.state_dict(), model_path)
