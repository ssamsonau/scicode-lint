import torch
import torch.nn as nn


class GANTrainer:
    def __init__(self, generator, discriminator, latent_dim=128):
        self.gen = generator
        self.disc = discriminator
        self.latent_dim = latent_dim
        self.opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, dataloader):
        for real_images, _ in dataloader:
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            self.opt_d.zero_grad()
            real_pred = self.disc(real_images)
            d_real_loss = self.criterion(real_pred, real_labels)

            z = torch.randn(batch_size, self.latent_dim)
            fake_images = self.gen(z)
            fake_pred = self.disc(fake_images.detach())
            d_fake_loss = self.criterion(fake_pred, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.opt_d.step()

            self.disc.eval()
            self.opt_g.zero_grad()
            gen_pred = self.disc(self.gen(z))
            g_loss = self.criterion(gen_pred, real_labels)
            g_loss.backward()
            self.opt_g.step()
