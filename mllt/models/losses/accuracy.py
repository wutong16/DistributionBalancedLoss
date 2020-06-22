import torch.nn as nn
import torch


def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    res = []
    mask = target >= 0
    for k in topk:
        _, idx = pred.topk(k, dim=1)
        pred_label = torch.zeros(target.size(), dtype=target.dtype, device=target.device).fill_(-2)
        pred_label = pred_label.scatter_(1, idx, 1)
        correct_k = pred_label.eq(target).sum().float()
        den = ((pred_label >= 0) * mask).sum()
        # den = (k * pred.size(0))
        if den > 0:
            prec = correct_k * 100.0 / den
        else:
            # if all are unknown label
            prec = torch.tensor(100.0)
        res.append(prec)
    return res[0] if return_single else res


class Accuracy(nn.Module):

    def __init__(self, topk=(1, )):
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        return accuracy(pred, target, self.topk)


if __name__ == '__main__':
    a = torch.tensor([[0.1,0.2,0.3],
                      [0.4,0.6,0.0]])
    b = torch.tensor([[1, 0, 1],
                      [1, 1, 0]])
    acc = accuracy(a,b)
    print(acc)