
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
import torchvision.models.resnet as resnet
import matplotlib.pyplot as plt
import time
import copy
import os


def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行10列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data.view_as(pred)).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, torch.tensor(len(labels)).cuda()  #返回正确的数量和这一次一共比较了多少元素

data_dir = 'data/data'
image_size = 224

# ToTensor使得H×W×C的ndarray变成C×H×W的tensor
# Normalize利用经验值：均值[0.485, 0.456, 0.406], 标准差[0.229, 0.224, 0.225]将图像各通道的数值转换至[-1, 1]的范围内
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                     transforms.Compose([
                                         transforms.RandomResizedCrop(image_size),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]))

val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                   transforms.Compose([
                                       transforms.Resize([256, 256]),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=True)

num_classes = len(train_dataset.classes)

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor

net = models.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
net = net.cuda() if use_cuda else net

for param in net.parameters():
    param.requires_grad = False

# net.fc.in_features是指原网络fc全连接层，输入特征的数量
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
net.fc = net.fc.cuda() if use_cuda else net.fc
for param in net.fc.parameters():
    param.requires_grad = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

record = []
num_epochs = 20
net.train(True)

#new income
for epoch in range(num_epochs):
    train_rights = []
    train_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.clone().detach().requires_grad_(True), target.clone().detach()
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = rightness(output, target)
        train_rights.append(right)
        loss = loss.cpu() if use_cuda else loss
        train_losses.append(loss.data.numpy())

    train_r = (sum([tup[0] for tup in train_rights]).cpu(), sum([tup[1] for tup in train_rights]).cpu())
    print('{}   {}'.format(train_r[0], train_r[1]))
pass
