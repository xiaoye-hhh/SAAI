import torch
import torch.nn as nn

class TransformerPool(nn.Module):
    def __init__(self, dim = 2048, part_num = 6, head_num = 8) -> None:
        super().__init__()

        self.part_num = part_num
        self.head_num = head_num
        self.scale = (dim // head_num) ** -0.5

        self.part_tokens = nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, head_num, part_num, dim//head_num), mode='fan_out'))
        self.pos_embeding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(18 * 9, dim), mode='fan_out'))
        self.kv = nn.Linear(dim, dim * 2)
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1) # [B, HW, C]
        x = x + self.pos_embeding

        kv = self.kv(x).reshape(B, H*W, 2, self.head_num, C//self.head_num).permute(2, 0, 3, 1, 4) # [2, B, head_num, HW, C//head_num]
        k, v = kv[0], kv[1] # [B, head_num, H*W, C//head_num]
        
        attn = self.part_tokens @ k.transpose(-1, -2) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, self.part_num, C) 

        return x.view(B, -1)


class SAFL(nn.Module):
    def __init__(self, dim = 2048, part_num = 6) -> None:
        super().__init__()

        self.part_num = part_num
        self.part_tokens = nn.Parameter(nn.init.kaiming_normal_(torch.empty(part_num, 2048)))
        self.pos_embeding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(18 * 9, dim)))

        self.active = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1) # [B, HW, C]
        x_pos = x + self.pos_embeding

        attn = self.part_tokens @ x_pos.transpose(-1, -2)
        attn = self.active(attn)

        x = attn @ x / H / W

        return x.view(B, -1), attn