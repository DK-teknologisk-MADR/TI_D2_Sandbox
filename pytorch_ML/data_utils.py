import torch
import os
import time
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms

def get_balanced_class_weights_from_dataset(dataset : Dataset, probs : np.array = None):
    '''
    Assumes dataset[i][1] is a integer label, if probs is none, classes will be evenly distributed. Otherwise, pfa
    '''
    targets = np.array([y for img, y in dataset])
    indices, counts = np.unique(targets, return_counts=True)
    if probs is None:
        probs = np.full(len(indices),1/len(indices),dtype=np.float)
    probs = probs / probs.sum()
    # checkout_imgs( tensor_pic_to_imshow_np(data_train[550][0]),'rgb')
    weights = probs / counts
    print("weights per class are: ",weights)
    weights = np.array([weights[data[1]] for data in dataset])
    return weights