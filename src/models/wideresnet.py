import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(nStages[0], nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._make_layer(nStages[1], nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._make_layer(nStages[2], nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.fc = nn.Linear(nStages[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, out_planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(WideBasic(in_planes, out_planes, dropout_rate, stride=s))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def wideresnet_26_10(num_classes: int, dropout_rate: float = 0.0):
    return WideResNet(depth=26, widen_factor=10, dropout_rate=dropout_rate, num_classes=num_classes)
