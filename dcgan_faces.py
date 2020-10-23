# dcgan_faces.py
#
#
# Author: Andrew Vu
# Date: 10/22/2020
#
# Train a GAN to create new celebrity faces using pictures of celebrities.
#
# DCGAN based on
# Implementation of PyTorch DCGAN tutorial by Nathan Inkawhich
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation
from IPython.display import HTML

# Seed
manualSeed = 411
manualSeed = random.randint(1, 10000);
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

###################
# Input Variables #
###################

# Dataset root directory
dataroot = "data/celeba"
# Worker threads
workers = 2
# Batch size
batch_size = 128
# Image size Default=64x64
image_size = 64
# Color Channels
nc = 3
# Length of latent vector
nz = 100
# Depth of feature maps in generator
ngf = 64
# Depth of feature maps in discriminator
ndf = 64
# Training epochs
num_epochs = 3
# Learning rate
lr = 0.0002
# beta1 hyperparameter for Adam optimizer Default = 0.5
beta1 = 0.5
# Nvidia GPUs available, 0 for CPU
ngpu = 0

########
# Data #
########

# Dataset
dataset = dataset.ImageFolder(root = dataroot,
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
)

# Dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = workers,
)

# Assign GPU/CPU to use
device = torch.device("cuda:0" if (torch.cuda.is_available and ngpu > 0) else "cpu")
real_batch=next(iter(dataloader))
pyplot.figure(figsize=(8,8))
pyplot.axis("off")
pyplot.title("Training Images")
pyplot.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],padding=2,normalize=True).cpu(),(1,2,0)))


######################
# Initialize Weights #
######################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.2)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#############
# Generator #
#############

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Instantiate Generator
netG=Generator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)
print(netG)


#################
# Discriminator #
#################

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Instantiate Discriminator
netD = Discriminator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
print(netD)

##################
# Loss Functions #
#       &        #
#   Optimizers   #
##################

# Initialize BCELoss
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

############
# Training #
############
img_list = []
G_losses = []
D_losses = []
iters = 0

# Training loop
print("Starting Training...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()

        #########################
        # Discriminator Network #
        #########################
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Pass real batch through Discriminator
        output = netD(real_cpu).view(-1)
        # Error and gradients for Discriminator in backwards spass
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generator produces fake image batch
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify fake image batch with Discriminator
        output = netD(fake.detach()).view(-1)
        # Discriminator error on fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for fake batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update Discriminator
        optimizerD.step()

        #####################
        # Generator Network #
        #####################
        netG.zero_grad()
        label.fill_(real_label)
        # Forward pass fake batch through Discriminator
        output = netD(fake).view(-1)
        # Error and gradients for Generator in forward pass
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update Generator
        optimizerG.step()

        # Training output
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Losses
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Generator's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


################
# Plot Results #
################

pyplot.figure(figsize=(10,5))
pyplot.title("Generator and Discriminator Training Loss")
pyplot.plot(G_losses,label="Generator")
pyplot.plot(D_losses,label="Discriminator")
pyplot.xlabel("Iterations")
pyplot.ylabel("Loss")
pyplot.legend()
pyplot.show()

fig = pyplot.figure(figsize=(8,8))
pyplot.axis("off")
ims = [[pyplot.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

##################
# Compare Images #
##################
real_batch = next(iter(dataloader))

# Real Images
pyplot.figure(figsize=(15,15))
pyplot.subplot(1,2,1)
pyplot.axis("off")
pyplot.title("Real Images")
pyplot.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Generated Images
pyplot.subplot(1,2,2)
pyplot.axis("off")
pyplot.title("Fake Images")
pyplot.imshow(np.transpose(img_list[-1],(1,2,0)))
pyplot.show()
