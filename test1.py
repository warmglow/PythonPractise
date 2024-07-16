
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

depth = [4, 8]


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, 5, padding = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding = 2)
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1], 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def retrieve_features(self, x):
        feature_map1 = F.relu(self.conv1(x))
        x = self.pool(feature_map1)
        feature_map2 = F.relu(self.conv2(x))
        return (feature_map1, feature_map2)


def rightness(output, target):
    preds = output.data.max(dim=1, keepdim=True)[1]
    return preds.eq(target.data.view_as(preds)).cpu(), len(target)


image_size = 28     # 图像单边像元尺寸
num_classes = 10    # 图像分类的标签种类
num_epochs = 20     # 训练循环数
batch_size = 64     # 一个batch样本数量

# root：存储图像数据的目录，train：True是训练数据，False是测试数据，transform：数据预处理，download：下载允许
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Dataloader通过调用dataset的__getitem__获取单个数据，用于生成data, target
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

indices = range(len(test_dataset))  # range函数得到的是一个名为range的类，属于iterable类型
indices_val = indices[:5000]        # 截取test前5000个样本作为验证集
indices_test = indices[5000:]       # 截取test后5000个样本作为测试集

sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

validation_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                shuffle=False, sampler=sampler_val)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                          shuffle=False, sampler=sampler_test)

# idx = 99
# muteimg = train_dataset[idx][0].numpy()
#
# plt.imshow(muteimg[0,...])
# plt.show()
# print('Label is ', train_dataset[idx][1])

net = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

record = []
weights = []

for epoch in range(num_epochs):
    train_rights = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.clone().requires_grad_(True), target.clone().detach()
        net.train()

        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = rightness(output, target)
        train_rights.append(right)

        if batch_idx % 100 == 0:
            net.eval()
            val_rights = []

            for (data, target) in validation_loader:
                data, target = data.clone().requires_grad_(True), target.clone().detach()
                output = net(data)
                right = rightness(output, target)
                val_rights.append(right)

            train_r = (sum(train_rights[0][0]).item(), train_rights[0][1])
            val_r = (sum(sum(tup[0]) for tup in val_rights).item(), sum(tup[1] for tup in val_rights))

            print('训练周期：{} [{}/{} ({:.0f}%)]\t, Loss：{:.6f}\t，训练准确率：{:.2f}%\t，校验准确率：{:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data,
                100. * train_r[0] / train_r[1],
                100. * val_r[0] / val_r[1]))
            record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))
            weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(),
                            net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])

            ## c'est ici.



