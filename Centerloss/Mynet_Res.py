import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 定义残差结构
class ResidualLayer(nn.Module):
    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            ConvolutionalLayer(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.sub_module(x)


# 定义卷积层
class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.sub_module(x)


# 定义主网络
class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.layer = nn.Sequential(
            # nn.Conv2d(1, 32, 5, 1, padding=2),  # 28--->28
            # nn.BatchNorm2d(32),
            # nn.PReLU(),
            ConvolutionalLayer(1, 32, 5, 1, 2),

            # nn.Conv2d(32, 32, 5, 1, padding=2),  # 28--->28
            # nn.BatchNorm2d(32),
            # nn.PReLU(),
            ConvolutionalLayer(32, 32, 5, 1, 2),

            nn.MaxPool2d(2, 2),  # 28--->14
            # 残差
            ResidualLayer(32),

            # nn.Conv2d(32, 64, 5, 1, padding=2),  # 14--->14
            # nn.BatchNorm2d(64),
            # nn.PReLU(),
            ConvolutionalLayer(32, 64, 5, 1, 2),

            # nn.Conv2d(64, 64, 5, 1, padding=2),  # 14--->14
            # nn.BatchNorm2d(64),
            # nn.PReLU(),
            ConvolutionalLayer(64, 64, 5, 1, 2),

            nn.MaxPool2d(2, 2),  # 14--->7
            # 残差
            ResidualLayer(64),

            # nn.Conv2d(64, 128, 5, 1, padding=2),  # 7--->7
            # nn.BatchNorm2d(128),
            # nn.PReLU(),
            ConvolutionalLayer(64, 128, 5, 1, 2),

            # nn.Conv2d(128, 128, 5, 1, padding=2),  # 7--->7
            # nn.BatchNorm2d(128),
            # nn.PReLU(),
            ConvolutionalLayer(128, 128, 5, 1, 2),

            nn.MaxPool2d(2, 2),  # 7--->3
            # 残差
            ResidualLayer(128),
        )

        self.feature = nn.Linear(128 * 3 * 3, 2)
        self.output = nn.Linear(2, 10)

    def forward(self, x):
        y_conv = self.layer(x)
        y_conv = y_conv.reshape(-1, 128 * 3 * 3)
        y_feature = self.feature(y_conv)  # (N 2)
        y_output = torch.log_softmax(self.output(y_feature), dim=1)  # (N 10)

        return y_feature, y_output

    # 画图（可视化）
    def visualize(self, features, labels, epoch):
        color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900',
                 '#009999']
        for i in range(10):
            plt.plot(features[labels == i, 0], features[labels == i, 1], '.', color=color[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc="upper right")
        plt.title("epochs=%d" % epoch)
        plt.savefig('./images/20-epoch=%d.jpg' % epoch)
