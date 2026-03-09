import torch


def train_gan_wrong(generator, discriminator, opt_g, opt_d, data_loader):
    for real_data in data_loader:
        fake_data = generator(torch.randn(32, 100))

        d_loss = discriminator(real_data).mean() - discriminator(fake_data).mean()
        opt_d.step()
        d_loss.backward()
        opt_d.zero_grad()

        g_loss = -discriminator(fake_data).mean()
        opt_g.step()
        g_loss.backward()
        opt_g.zero_grad()
