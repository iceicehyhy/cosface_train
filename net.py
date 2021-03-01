#!usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class sphere20(nn.Module):
    def __init__(self):
        super(sphere20, self).__init__()

        # input: batch_size, channel_num, pic_width, pic_height (B, 3, 112, 112)

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)  # => b*64*56*56
        self.relu1_1 = nn.ReLU(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)  # => b*64*56*56
        self.relu1_2 = nn.ReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)   # => b*64*56*56
        self.relu1_3 = nn.ReLU(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1)  # => b*128*28*28
        self.relu2_1 = nn.ReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_2 = nn.ReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_3 = nn.ReLU(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1)  # => b*256*14*14
        self.relu3_1 = nn.ReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_2 = nn.ReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_3 = nn.ReLU(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1)  # => b*512*7*7
        self.relu4_1 = nn.ReLU(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.relu4_2 = nn.ReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.relu4_3 = nn.ReLU(512)

        self.fc4 = nn.Linear(512 * 7 * 7, 512)

        # weight: initialization
        for m in self.modules():
            # 如果是 卷积层 或者 Linear 层
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0), -1)
        x = self.fc4(x)

        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)










