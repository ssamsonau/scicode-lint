import torch


def train_gan(generator, discriminator, opt_g, opt_d, data_loader):
    for real_data in data_loader:
        opt_d.zero_grad()
        fake_data = generator(torch.randn(32, 100)).detach()
        d_loss = discriminator(real_data).mean() - discriminator(fake_data).mean()
        d_loss.backward()
        opt_d.step()

        opt_g.zero_grad()
        fake_data = generator(torch.randn(32, 100))
        g_loss = -discriminator(fake_data).mean()
        g_loss.backward()
        opt_g.step()
