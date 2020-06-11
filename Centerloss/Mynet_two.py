import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, padding=2),  # 28--->28
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.Conv2d(32, 32, 5, 1, padding=2),  # 28--->28
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 28--->14

            nn.Conv2d(32, 64, 5, 1, padding=2),  # 14--->14
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, 5, 1, padding=2),  # 14--->14
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 14--->7

            nn.Conv2d(64, 128, 5, 1, padding=2),  # 7--->7
            nn.BatchNorm2d(128),
            nn.PReLU(),

            nn.Conv2d(128, 128, 5, 1, padding=2),  # 7--->7
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 7--->3
        )
        self.feature = nn.Linear(128 * 3 * 3, 2)
        self.output = nn.Linear(2, 10)

    def forward(self, x):
        y_conv = self.layer(x)
        y_conv = y_conv.reshape(-1, 128 * 3 * 3)
        y_feature = self.feature(y_conv)  # (n 2)
        y_output = torch.log_softmax(self.output(y_feature), dim=1)  # (n 10)
        return y_feature, y_output

    def visualize(self, features, labels, epoch):
        # plt.ion()
        plt.clf()
        color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900',
                 '#009999']
        for i in range(10):
            plt.plot(features[labels == i, 0], features[labels == i, 1], '.', color=color[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc="upper right")
        plt.title("epochs=%d" % epoch)
        plt.savefig('./images1/10000-epoch=%d.jpg' % epoch)
