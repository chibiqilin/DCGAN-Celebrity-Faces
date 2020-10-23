# DCGAN-Celebrity-Faces
Implemented DCGAN based on PyTorch example for Face Generation using Celeb-A Faces dataset

# What is a DCGAN?
Deep Convolutional Generative Adversarial Networks are a type of GAN described by [Radford et. al](https://arxiv.org/abs/1511.06434) which make use of convolutional layers.

# Input
Input consists of 3x64x64 color images found in the [Celeb-A Faces Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or via [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg).

# Variables
Input variables:
*dataroot - path to dataset, default is read from /data/celeba/
*workers - number of worker threads for loading the data
*batch_size - training batch size
*image_size - image size, any changes should be reflected in discriminator and generator structure
*nc - number of color channels, color images = 3
*nz - length of latent vector
*ngf - depth of feature maps carried through the generator
*ndf - depth of feature maps propagated through the discriminator
*num_epochs - number of epochs for training
*lr - learning rate, Radford et. al suggests lr = 0.0002
*beta1 - beta1 hyperparameter for Adam optimizers. Radford et. al suggests beta1 = 0.5
*ngpu - number of GPUs, 0 = CPU.

# Credit
Credit to Nathan Inkawhich for the [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
