from Centerloss.Mynet_Res import MainNet
from Centerloss.centerloss import Centerloss
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as trans
from torch.utils.data import DataLoader
import os


class TRAIN:
    def training(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(mean=(0.1306,), std=(0.3081,))
        ])
        savepath = "models/centerlossnet_Resnet.pth"

        traindata = torchvision.datasets.MNIST(root="./MNIST", download=True, train=True, transform=transform)
        trainloader = DataLoader(traindata, shuffle=True, batch_size=200, num_workers=4)

        # data = next(iter(trainloader))[0]   # 通过iter(),迭代loader。就一个元素，迭代一次，通过[0]取出data来。
        # print(data.shape)
        # mean = torch.mean(data, dim=(0, 2, 3))   # 求均值，通过dim()给轴，NCHW,
        # std = torch.std(data, dim=(0, 2, 3))   # 求标准差，通过dim()给轴
        # print(mean, std)

        net = MainNet().to(device)
        centerloss = Centerloss().to(device)

        if os.path.exists(savepath):
            net.load_state_dict(torch.load(savepath, map_location=device))
        else:
            print("No param!")
        cls_lossfunc = nn.NLLLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001)
        epoch = 140
        while True:
            print("当前训练次数epoch=", epoch)
            featureloader = []
            labelloader = []
            for i, (x, y) in enumerate(trainloader):
                x, y = x.to(device), y.to(device)
                feature, output = net(x)

                cls_loss = cls_lossfunc(output, y)
                y = y.float()
                center_loss = centerloss.forward(feature, y, 2)
                loss = cls_loss + center_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                featureloader.append(feature)
                labelloader.append(y)
                if i % 100 == 0:
                    print("{}/{} | cls_loss={} | center_loss={} | loss={} ".format(i, len(trainloader), cls_loss,
                                                                                   center_loss, loss))
            features = torch.cat(featureloader, 0)
            labels = torch.cat(labelloader, 0)
            net.visualize(features.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
            epoch += 1
            torch.save(net.state_dict(), savepath)
            if epoch == 151:
                break


if __name__ == '__main__':
    train = TRAIN()
    train.training()
