import torch
import torch.nn as nn
import torch.nn.functional as F

import network.resnet38d


class Net(network.resnet38d.Net):
    def __init__(self, num_classes):
        super().__init__()

        self.fc8_seg_conv1 = nn.Conv2d(4096, 512, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv1.weight)

        self.fc8_seg_conv2 = nn.Conv2d(512, num_classes, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv2.weight)

        self.from_scratch_layers = [self.fc8_seg_conv1, self.fc8_seg_conv2]


    def forward(self, x):
        x = super().forward(x)
        x_seg = F.relu(self.fc8_seg_conv1(x))
        x_seg = self.fc8_seg_conv2(x_seg)

        return x_seg


    def get_10x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' in name:
                yield param

    def get_1x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' not in name:
                yield param

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups