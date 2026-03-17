import torch
import torch.nn as nn


def train_gan_step(generator, discriminator, opt_g, opt_d, real_data, latent_dim=64):
    batch_size = real_data.size(0)

    opt_d.zero_grad()
    real_pred = discriminator(real_data)
    d_real_loss = nn.functional.binary_cross_entropy_with_logits(
        real_pred, torch.ones_like(real_pred)
    )
    z = torch.randn(batch_size, latent_dim)
    fake_data = generator(z).detach()
    fake_pred = discriminator(fake_data)
    d_fake_loss = nn.functional.binary_cross_entropy_with_logits(
        fake_pred, torch.zeros_like(fake_pred)
    )
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    opt_d.step()

    opt_g.zero_grad()
    z = torch.randn(batch_size, latent_dim)
    gen_pred = discriminator(generator(z))
    g_loss = nn.functional.binary_cross_entropy_with_logits(gen_pred, torch.ones_like(gen_pred))
    g_loss.backward()
    opt_g.step()

    return d_loss.item(), g_loss.item()
