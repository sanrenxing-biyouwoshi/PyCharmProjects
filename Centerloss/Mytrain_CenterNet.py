from CenterLoss.Mynet_CenterNet import SoftmaxNET, CenterNET
from CenterLoss.centerloss import centerloss
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as trans
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os


class TRAIN:
    def training(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        savepath1 = "models/featurelossnet100.pth"
        savepath2 = "models/softmaxlossnet100.pth"
        transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(mean=(0.1306,), std=(0.3081,))
        ])

        traindata = torchvision.datasets.MNIST(root="./MNIST", download=True, train=True, transform=transform)
        trainloader = DataLoader(traindata, shuffle=True, batch_size=100, num_workers=4)

        feature_net = CenterNET().to(device)
        output_net = SoftmaxNET().to(device)

        if os.path.exists(savepath1):
            feature_net.load_state_dict(torch.load(savepath1))
        else:
            print("No param!")
        if os.path.exists(savepath2):
            output_net.load_state_dict(torch.load(savepath2))
        else:
            print("No param!")

        cls_lossfunc = nn.NLLLoss()
        optimizer1 = optim.SGD(feature_net.parameters(), lr=0.01, momentum=0.9)
        optimizer2 = optim.SGD(output_net.parameters(), lr=0.001, momentum=0.9)

        epoch = 1
        while True:
            print("当前训练次数epoch=", epoch)
            featureloader = []
            labelloader = []
            for i, (x, y) in enumerate(trainloader):
                x, y = x.to(device), y.to(device)
                feature = feature_net(x)
                output = output_net(feature)

                feature_loss = centerloss(feature, y.float(), lambdas=2)
                cls_loss = cls_lossfunc(output, y.long())

                optimizer1.zero_grad()
                feature_loss.backward(retain_graph=True)
                optimizer1.step()

                optimizer2.zero_grad()
                cls_loss.backward()
                optimizer2.step()

                featureloader.append(feature)
                labelloader.append(y)
                if i % 100 == 0:
                    print("{}/{} | feature_loss={} | cls_loss={}".format(i, len(trainloader), feature_loss, cls_loss))

            features = torch.cat(featureloader, 0).data.cpu().numpy()
            labels = torch.cat(labelloader, 0).data.cpu().numpy()
            CenterNET.DRAW(features, labels.data, epoch)

            epoch += 1
            torch.save(feature_net.state_dict(), savepath1)
            torch.save(output_net.state_dict(), savepath2)
            if epoch == 51:
                break


if __name__ == '__main__':
    train = TRAIN()
    train.training()
