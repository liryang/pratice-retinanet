import torch
import torch.nn as nn

from fpn import FPN50
from torch.autograd import Variable
import math


class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes, firstinit=False):
        super(RetinaNet, self).__init__()
        self.name = 'RetinaNet'
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)
        self.best_loss = None
        self.lr = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        self.fpn = FPN50(firstinit)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
            # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
            # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)


def test():
    model = RetinaNet(2, firstinit=True)
    loc_preds, cls_preds = model(Variable(torch.randn(3, 3, 256, 256)))
    print(loc_preds.size())
    print(cls_preds.size())
    # loc_grads = torch.randn(loc_preds.size())
    # loc_preds.backward(loc_grads)
    # cls_grads = torch.randn(cls_preds.size())
    # cls_preds.backward(cls_grads)

test()
