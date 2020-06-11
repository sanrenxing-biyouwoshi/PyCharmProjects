from Mynet import NET
from loss import centerloss
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as trans
from torch.utils.data import DataLoader
import os


class TRAIN:
    def training(self):
        savepath = "models/centerlossnet2.pth"
        transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(mean=(0.5,), std=(0.5,))
        ])
        traindata = torchvision.datasets.MNIST(root="./MNIST", download=True, train=True,transform=transform)
        trainloader = DataLoader(traindata, shuffle=True, batch_size=6000, num_workers=4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = NET().to(device)
        if os.path.exists(savepath):
            net.load_state_dict(torch.load(savepath))
        else:
            print("No param!")
        cls_lossfunc = nn.NLLLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.0005)
        epoch = 200
        while True:
            print("当前训练次数epoch=", epoch)
            featureloader = []
            labelloader = []
            for i, (x, y) in enumerate(trainloader):
                x, y = x.to(device), y.to(device)
                feature, output = net(x)

                cls_loss = cls_lossfunc(output, y)
                y = y.float()
                center_loss = centerloss(feature, y, 2)
                loss = cls_loss + center_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                featureloader.append(feature)
                labelloader.append(y)
                if i % 10 == 0:
                    print("{}/{},{}, {}, {} ".format(i, len(traindata), cls_loss, center_loss, loss))
            features = torch.cat(featureloader, 0)
            labels = torch.cat(labelloader, 0)
            #net.visualize(features.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
            epoch += 1
            torch.save(net.state_dict(), savepath)
            if epoch == 2000:
                break


if __name__ == '__main__':
    train = TRAIN()
    train.training()
