'''pretty much taken from the torchvision resnet model'''

import torch.nn as nn
import torch.nn.functional as F

def build_entry_flow(args):
    net = nn.Sequential(nn.Conv2d(3, 32, 1),
                        nn.ReLU(),
                        nn.Conv2d(32, 3, 1))
    return net


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, args):
        super(ResNet18, self).__init__()
        self.ch = eval(args['model']['cond_net_channels'])
        self.res_lev = len(eval(args['model']['inn_coupling_blocks']))
        self.in_planes = self.ch

        self.conv1 = nn.Conv2d(3, self.ch[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ch[0])
        self.relu = nn.LeakyReLU()
        self.preprocess_level = nn.Sequential(self.conv1, self.bn1, self.relu)

        self.layers = {}
        in_planes = self.ch[0]

        for i in range(self.res_lev):
            if i == 0:
                self.layers['layer'+ str(i)] = self._make_layer(BasicBlock, in_planes, self.ch[i],
                                                                num_blocks=2, stride=1)
            else:
                self.layers['layer' + str(i)] = self._make_layer(BasicBlock, in_planes, self.ch[i],
                                                                 num_blocks=2, stride=2)
            in_planes = self.ch[i]

        # make resolution layers automatically depending on default/conf.ini
        self.resolution_levels = nn.ModuleList([self.layers['layer'+str(i)] for i in range(self.res_lev)])

    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        layers = [block(in_planes, planes, stride)]
        for i in range(num_blocks - 1):
            layers.append(block(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = [x]
        for i, layer in enumerate(self.resolution_levels):
            if i == 0:
                x = self.preprocess_level(x)
                outputs.append(layer(x))
            else:
                outputs.append(layer(outputs[-1]))
        return outputs[1:]
