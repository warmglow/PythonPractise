
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import copy

depth = [4, 8]
class Transfer(nn.Module):
    def __init__(self):
        super(Transfer, self).__init__()
        self.net1_conv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.net_pool = nn.MaxPool2d(2, 2)
        self.net1_conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)

        self.net2_conv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.net2_conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)

        self.fc1 = nn.Linear(2 * image_size // 4 * image_size // 4 * depth[1], 1024)
        self.fc2 = nn.Linear(1024, 2 * num_classes)
        self.fc3 = nn.Linear(2 * num_classes, num_classes)
        self.fc4 = nn.Linear(num_classes, 1)

    def foward(self, x, y, training = True):
        x = F.relu(self.net1_conv1(x))
        x = self.net_pool(x)
        x = F.relu(self.net1_conv2(x))
        x = self.net_pool(x)
        x = x.view




image_size = 28
num_classes = 10
num_epochs = 20
batch_size = 64

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor

train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

sampler1 = torch.utils.data.sampler.SubsetRandomSampler(np.random.permutation(range(len(train_dataset))))
sampler2 = torch.utils.data.sampler.SubsetRandomSampler(np.random.permutation(range(len(train_dataset))))

train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, shuffle=False, sampler=sampler1)
train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, shuffle=False, sampler=sampler2)

val_size = 5000
val_indices1 = range(val_size)
val_indices2 = np.random.permutation(range(val_size))

test_indices1 = range(val_size, len(test_dataset))
test_indices2 = np.random.permutation(test_indices1)

val_sampler1 = torch.utils.data.sampler.SubsetRandomSampler(val_indices1)
val_sampler2 = torch.utils.data.sampler.SubsetRandomSampler(val_indices2)

test_sampler1 = torch.utils.data.sampler.SubsetRandomSampler(test_indices1)
test_sampler2 = torch.utils.data.sampler.SubsetRandomSampler(test_indices2)

val_loader1 = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                          shuffle=False, sampler=val_sampler1)
val_loader2 = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                          shuffle=False, sampler=val_sampler2)
test_loader1 = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                           shuffle=False, sampler=test_sampler1)
test_loader2 = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                           shuffle=False, sampler=test_sampler2)
