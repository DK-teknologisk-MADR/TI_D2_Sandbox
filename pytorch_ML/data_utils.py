import torch
import os
import time
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
transform = transforms.Compose(
    [transforms.ToTensor()])
dataset = CIFAR10(root="\somewhere",download=True,transform=transform)
dataset[0][0].sum(axis=(1,2))
