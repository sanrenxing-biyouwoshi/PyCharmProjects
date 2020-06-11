import torch


def center_loss():
    data = torch.tensor([[2, 3], [4, 5], [6, 7], [8, 9], [4, 7]])
    lable = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32)  # 矩阵操作一般是浮点数
    center = torch.tensor([[1, 1], [2, 2]], dtype=torch.float32)    # 随机给的2个中心

    center_exp = center.index_select(0, lable.long())  # 按照lable的方式取center,类型必须为long。
    # print(center_exp)
    # histc(数据，类别个数，最小值，最大值)    # tensor([3., 2.])
    count = torch.histc(lable, int(max(lable).item() + 1), int(min(lable).item()), int(max(lable).item()))
    print(count)
    count_exp = count.index_select(0, lable.long())  # tensor([3., 3., 2., 3., 2.])

    # loss = torch.pow(data - center_exp, 2)
    # loss = torch.sum(torch.pow(data - center_exp, 2), dim=1)
    # loss = torch.div(torch.sum(torch.pow(data - center_exp, 2), dim=1), count_exp)
    # print(loss)
    loss = torch.mean(torch.div(torch.sum(torch.pow(data - center_exp, 2), dim=1), count_exp))
    return loss


if __name__ == '__main__':
    print(center_loss())
