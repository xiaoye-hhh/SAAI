import torch.nn as nn
import torch

class ContextBlock2d(nn.Module):

    def __init__(self, dim):
        super(ContextBlock2d, self).__init__()
        
        self.conv_mask = nn.Conv2d(dim, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(dim, dim//16, kernel_size=1),
                nn.LayerNorm([dim//16, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim//16, dim, kernel_size=1)
        )

        self.IN = nn.InstanceNorm2d(dim)


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        # 汇集全文的信息 对应的像素点进行匹配，整个图像的像素点全部相加
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        # [N, C, 1, 1]
        mask = torch.sigmoid(self.channel_mul_conv(context))
        out = x * mask + (1 - mask) * self.IN(x)

        return out