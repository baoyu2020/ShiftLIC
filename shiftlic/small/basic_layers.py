import sys
import torch
from torch import nn as nn
from torch.nn import functional as F

sys.path.append("..")
from utils import default_init_weights

from torch import Tensor
from compressai.layers import GDN

    
class Shift4(nn.Module):  # 四个组的方向（上下左右）
    def __init__(self, groups=4, stride=1, mode='constant') -> None:
        super().__init__()
        self.g = groups
        self.mode = mode
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.shape
        out = torch.zeros_like(x)

        pad_x = F.pad(x, pad=[self.stride for _ in range(4)], mode=self.mode)
        assert c == self.g * 4

        cx, cy = self.stride, self.stride
        stride = self.stride
        out[:,0*self.g:1*self.g, :, :] = pad_x[:, 0*self.g:1*self.g, cx-stride:cx-stride+h, cy:cy+w]
        out[:,1*self.g:2*self.g, :, :] = pad_x[:, 1*self.g:2*self.g, cx+stride:cx+stride+h, cy:cy+w]
        out[:,2*self.g:3*self.g, :, :] = pad_x[:, 2*self.g:3*self.g, cx:cx+h, cy-stride:cy-stride+w]
        out[:,3*self.g:4*self.g, :, :] = pad_x[:, 3*self.g:4*self.g, cx:cx+h, cy+stride:cy+stride+w]
        return out
    

class ResidualBlockShift(nn.Module):
    def __init__(self, in_feat, out_feat, res_scale=1, pytorch_init=False):
        super(ResidualBlockShift, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_feat, in_feat, kernel_size=1)
        self.conv2 = nn.Conv2d(in_feat, out_feat, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.shift = Shift4(groups=in_feat//4, stride=1)   # 只需要在这里更改groups与stride

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

        if in_feat != out_feat:
            self.skip = nn.Conv2d(in_feat, out_feat, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x): 

        identity = self.skip(x)
        out = self.conv2(self.shift(self.relu(self.conv1(x))))

        return identity + out * self.res_scale


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


#### Cheap Spatial-Channel Attention
class CheapChannelV1(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])
                 
        # Activation
        self.act = nn.GELU() 

        self.fusion1 = nn.Conv2d(chunk_dim*2, chunk_dim*2, 1)
        self.fusion2 = nn.Conv2d(chunk_dim*3, chunk_dim*3, 1)
        self.fusion3 = nn.Conv2d(chunk_dim*4, chunk_dim*4, 1)

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        s = []
        # s = [torch.Tensor()] * self.n_levels
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**i, w//2**i)
                t = F.adaptive_max_pool2d(xc[i], p_size)
                t = self.mfr[i](t)
                t = F.interpolate(t, size=(h, w), mode='nearest')                
            else:
                # print(xc[i].shape, i)
                # s.append(self.mfr[i](xc[i])) #S0
                t = self.mfr[i](xc[i])
            s.append(t)
                  
        #  融合1
        res1 = self.fusion1(channel_shuffle(torch.cat([s[0], s[1]],dim=1), 8))   # 只需要更改这里groups
        res2 = self.fusion2(channel_shuffle(torch.cat([res1, s[2]],dim=1), 8))
        res3 = self.fusion3(channel_shuffle(torch.cat([res2, s[3]],dim=1), 8))

        out = self.act(res3) * x
        return out
    

class CheapCS1(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()

        self.CheapChannel = CheapChannelV1(dim)
        self.CheapSpatial = nn.Sequential(
            ResidualBlockShift(dim, dim*2),
            nn.GELU(),
            nn.Conv2d(dim*2, dim, 1, bias=False),
        )
    
    def forward(self, x):
        y = self.CheapChannel(x) + x
        y = self.CheapSpatial(y) + y
        return y
