import numpy as np
import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F

def makeGaussian(size, fwhm = 3):
    """ 生成高斯核 """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

class SaliencySampler(nn.Module):
    def __init__(self, task_input_size):
        super(SaliencySampler, self).__init__()
        
        self.grid_size = 31
        self.padding_size = 30
        self.global_size = self.grid_size+2*self.padding_size
        self.input_size_net = task_input_size
        self.gaussian_weights = torch.tensor(makeGaussian(2*self.padding_size+1, fwhm = 13), dtype=torch.float, device='cuda').reshape(1, 1, 2*self.padding_size+1, 2*self.padding_size+1)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.MaxPool2d(2),
            *(list(models.resnet18(pretrained=True).children())[:-3]),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1,padding=0,stride=1),
            nn.Upsample(size=(self.grid_size, self.grid_size), mode='bilinear')
        )

        self.P_basis = torch.zeros(2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size).cuda()
        for k in range(2):
            for i in range(self.global_size):
                for j in range(self.global_size):
                    self.P_basis[k,i,j] = k*(i-self.padding_size)/(self.grid_size-1.0)+(1.0-k)*(j-self.padding_size)/(self.grid_size-1.0)

    def create_grid(self, x):
        P = self.P_basis.expand(x.shape[0], 2, self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size)
        
        x_cat = torch.cat((x,x),1)
        p_filter = F.conv2d(x, self.gaussian_weights)
        x_mul = torch.mul(P, x_cat).view(-1, 1, self.global_size,self.global_size)
        all_filter =  F.conv2d(x_mul, self.gaussian_weights).view(-1,2,self.grid_size,self.grid_size)

        x_filter = all_filter[:,0,:,:].contiguous().view(-1,1,self.grid_size,self.grid_size)
        y_filter = all_filter[:,1,:,:].contiguous().view(-1,1,self.grid_size,self.grid_size)

        x_filter = x_filter/p_filter
        y_filter = y_filter/p_filter

        xgrids = x_filter*2-1
        ygrids = y_filter*2-1
        xgrids = torch.clamp(xgrids,min=-1,max=1)
        ygrids = torch.clamp(ygrids,min=-1,max=1)

        xgrids = xgrids.view(-1,1,self.grid_size,self.grid_size)
        ygrids = ygrids.view(-1,1,self.grid_size,self.grid_size)

        grid = torch.cat((xgrids,ygrids),1)
        grid = nn.Upsample(size=self.input_size_net, mode='bilinear')(grid)

        grid = torch.transpose(grid,1,2)
        grid = torch.transpose(grid,2,3)

        return grid

    def forward(self, x):
        # 得到显著性图
        xs = self.localization(x)

        # 归一化
        xs = xs.view(-1, self.grid_size*self.grid_size)
        xs = xs.softmax(1)
        xs = xs.view(-1, 1, self.grid_size, self.grid_size)
        xs_hm = nn.ReplicationPad2d(self.padding_size)(xs)

        grid = self.create_grid(xs_hm)
        return F.grid_sample(x, grid)
