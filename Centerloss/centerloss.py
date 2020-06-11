import torch
import torch.nn as nn


# def centerloss(feature, label, lambdas):
#     label = label.unsqueeze(0)
#     center = nn.Parameter(torch.randn(label.shape[1], feature.shape[1]), requires_grad=True).cuda()
#     label = label.squeeze()
#     center_exp = center.index_select(0, label.long())
#     count = torch.histc(label, int(max(label).item() + 1), 0, int(max(label).item()))
#     count_exp = count.index_select(0, label.long())
#     loss = lambdas / 2 * (torch.mean(torch.div(torch.sum(torch.pow(feature - center_exp, 2), dim=1), count_exp)))
#     return loss


class Centerloss(nn.Module):

    def __init__(self):
        super(Centerloss, self).__init__()
        self.center = nn.Parameter(torch.randn(200, 2), requires_grad=True)

    def forward(self, feature, label, lambdas):
        center_exp = self.center.index_select(0, label.long())
        count = torch.histc(label, int(max(label).item() + 1), 0, int(max(label).item()))
        count_exp = count.index_select(0, label.long())
        loss = lambdas / 2 * (torch.mean(torch.div(torch.sum(torch.pow(feature - center_exp, 2), dim=1), count_exp)))
        return loss


if __name__ == '__main__':
    # Centerloss()
    data = torch.tensor([[2, 3], [4, 5], [6, 7], [8, 9], [4, 7]]).cuda()
    lable = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32).cuda()  # 矩阵操作一般是浮点数
    loss = Centerloss().forward(data, lable, 2)
    print(loss)
