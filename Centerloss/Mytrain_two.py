from Centerloss.Mynet_two import NET
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
        savepath = "models/CNET10000.pth"
        transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(mean=(0.1306,), std=(0.3081,))
        ])
        traindata = torchvision.datasets.MNIST(root="./MNIST", download=True, train=True, transform=transform)
        trainloader = DataLoader(traindata, shuffle=True, batch_size=600, num_workers=4)

        net = NET().to(device)
        center = Centerloss().to(device)
        if os.path.exists(savepath):
            net.load_state_dict(torch.load(savepath))
        else:
            print("No param!")

        cls_lossfunc = nn.NLLLoss()
        optimizer1 = optim.SGD(net.parameters(), lr=0.005)
        optimizer2 = optim.SGD(center.parameters(), lr=0.001)
        epoch = 99
        while True:
            print("当前训练次数epoch=", epoch)
            featureloader = []
            labelloader = []
            for i, (x, y) in enumerate(trainloader):
                x, y = x.to(device), y.to(device)
                feature, output = net(x)
                cls_loss = cls_lossfunc(output, y.long())
                center_loss = center.forward(feature, y.float(), 2)
                loss = cls_loss + center_loss

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()

                featureloader.append(feature)
                labelloader.append(y)
                if i % 100 == 0:
                    print("{}/{} | cls_loss={} | center_loss={}".format(i, len(trainloader), cls_loss, center_loss))
            features = torch.cat(featureloader, 0)
            labels = torch.cat(labelloader, 0)
            net.visualize(features.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
            epoch += 1
            torch.save(net.state_dict(), savepath)
            if epoch == 101:
                break


if __name__ == '__main__':
    train = TRAIN()
    train.training()
