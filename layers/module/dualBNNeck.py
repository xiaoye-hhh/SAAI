from torch import nn 

class DualBNNeck(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.bn_neck_v = nn.BatchNorm1d(dim)
        self.bn_neck_i = nn.BatchNorm1d(dim)
        nn.init.constant_(self.bn_neck_i.bias, 0) 
        nn.init.constant_(self.bn_neck_v.bias, 0) 
        self.bn_neck_v.bias.requires_grad_(False)
        self.bn_neck_i.bias.requires_grad_(False)

    def forward(self, x, sub):
        mask_i = sub == 1
        mask_v = sub == 0

        x[mask_i] = self.bn_neck_i(x[mask_i])
        x[mask_v] = self.bn_neck_v(x[mask_v])

        return x
