#!/usr/bin/env python3


import torchgan
from torchgan.models import *
from torchgan.trainer import Trainer
from torchgan.losses import *
from torch import nn


# declare the gan network
# # declare the gan network
PATCH_SIZE = 96 # choose the desired size for generated output of GAN
dcgan_network = {
    "generator": {
        "name": DCGANGenerator,
        "args": {
            "encoding_dims": 128,
            "out_channels": 3,
            "out_size": PATCH_SIZE,  # need to change this to 128 in case
            "step_channels": 64,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.Tanh(),
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": DCGANDiscriminator,
        "args": {
            "in_channels": 3,
            "in_size": PATCH_SIZE,
            "step_channels": 64,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.LeakyReLU(0.2),
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}
wgangp_losses = [
    WassersteinGeneratorLoss(),
    WassersteinDiscriminatorLoss(),
    WassersteinGradientPenalty(),
]


trainer = Trainer(
    dcgan_network, wgangp_losses, sample_size=96, epochs=1000, device=device
)

model_path = './gan4.model'
trainer.load_model(model_path)
netG = trainer.generator
netG.eval()
