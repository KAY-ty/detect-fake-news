# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from dct_transform import *
import data_loader
import FrequencyDomain
import PixelDomain
import Fusion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    train_set, val_set, test_set = data_loader.get_datasets()

    f_net = FrequencyDomain.FrequencyDomain().to(device)
    p_net = PixelDomain.PixelDomain().to(device)
    net = Fusion.Fusion(p_net,f_net).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    #batch print threshold
    print_count = 1

    # 随便找了个数据集测试模型是否维度匹配, 先保证跑通过模型的forward函数
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        accuracy = 0.0
        for i, data in enumerate(train_set, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, dct_imgs, labels = data

            # get the pictures' addresses
            # pics_address = [i[0] for i in train_set.dataset.samples[i*64:(i+1)*64]]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs, dct_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            accuracy += (outputs.argmax(1) == labels).sum() / float(len(labels))

            if i % print_count == print_count-1:    # print every x mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_count))
                running_loss = 0.0
                print('[%d, %5d] accuracy: %.3f' %
                      (epoch + 1, i + 1, accuracy / print_count))
                accuracy = 0.0

    print('Finished Training')


