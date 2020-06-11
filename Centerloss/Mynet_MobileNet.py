import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# MobileNet_V2_1
class MobileNet(nn.Module):
    def __init__(self, in_channels):
        super(MobileNet, self).__init__()
        self.MobileNet_V2_1_layer = nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels * 2, 1, 1),
            ConvolutionalLayer(in_channels * 2, in_channels * 2, 3, 1, padding=1, groups=in_channels * 2),
            ConvolutionalLayer(in_channels * 2, in_channels, 1, 1),
            nn.MaxPool2d(2, 2),

        )

    def forward(self, x):
        return self.MobileNet_V2_1_layer(x)


# 定义卷积层
class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False, groups=1):
        super(ConvolutionalLayer, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, groups=groups),
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
            ConvolutionalLayer(1, 32, 5, 1, padding=2),
            ConvolutionalLayer(32, 32, 5, 1, padding=2),
            MobileNet(32),
            ConvolutionalLayer(32, 64, 5, 1, padding=2),
            MobileNet(64),
            ConvolutionalLayer(64, 128, 5, 1, padding=2),
            MobileNet(128)
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
        plt.clf()
        color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900',
                 '#009999']
        for i in range(10):
            plt.plot(features[labels == i, 0], features[labels == i, 1], '.', color=color[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc="upper right")
        plt.title("epochs=%d" % epoch)
        plt.savefig('./images1/epoch=%d.jpg' % epoch)
