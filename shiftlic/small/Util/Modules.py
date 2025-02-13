import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=2000):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # x(B,N,d)  pos_table(1,N,d)
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or (self.head_dim ** -0.5)

        # q,k的输出维度一定相同，可以与v的维度不同，最后通过全连接层得到相同维度dim
        self._to_q = nn.Linear(dim, dim, bias=qkv_bias)  
        self._to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self._to_v = nn.Linear(dim, dim, bias=qkv_bias)  
        self.drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, has_mask=False):
        B, N, C = x.shape
        q = self._to_q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self._to_k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self._to_v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if has_mask:
            device = x.device
            mask = (torch.triu(torch.ones(N, N)) == 1).transpose(0, 1).to(device)  # shape[N, N] Ture or False
            attn = attn.masked_fill(mask == 0, float('-inf'))
            # print(attn)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, d_in, d_hid, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)  # position-wise
        self.fc2 = nn.Linear(d_hid, d_in)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, d_hid, qkv_bias=False, qk_scale=None, drop=0.1):
        super().__init__()
        self.self_attn = Attention(dim, num_heads, qkv_bias, qk_scale, drop=drop)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, d_hid, drop=drop)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class EncoderLayer_mask(nn.Module):
    def __init__(self, dim, num_heads, d_hid, qkv_bias=False, qk_scale=None, drop=0.1):
        super().__init__()
        self.self_attn = Attention(dim, num_heads, qkv_bias, qk_scale, drop=drop)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, d_hid, drop=drop)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x, has_mask=True):
        x = x + self.self_attn(self.layer_norm1(x), has_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class EncoderLayers(nn.Module):
    def __init__(self, layer_num, emb_dim, nheads):
        super().__init__()
        self.enc_layer = EncoderLayer(dim=emb_dim, num_heads=nheads, d_hid=emb_dim * 4)
        self.layer_num = layer_num
        self.layers = nn.ModuleList([self.enc_layer for _ in range(layer_num)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EncoderLayer_noLN(nn.Module):
    def __init__(self, dim, num_heads, d_hid, qkv_bias=False, qk_scale=None, drop=0.1):
        super().__init__()
        self.self_attn = Attention(dim, num_heads, qkv_bias, qk_scale, drop=drop)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, d_hid, drop=drop)
        # self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(x)
        return x

class EncoderLayer_Mlp(nn.Module):
    def __init__(self, dim, d_hid, drop=0.1):
        super().__init__()
        # self.self_attn = Attention(dim, num_heads, qkv_bias, qk_scale, drop=drop)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, d_hid, act_layer=nn.ReLU, drop=drop)
        # self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm(x))
        return x

class EncoderLayer_Attn(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.1):
        super().__init__()
        self.self_attn = Attention(dim, num_heads, qkv_bias, qk_scale, drop=drop)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        # self.mlp = Mlp(dim, d_hid, act_layer=nn.ReLU, drop=drop)
        # self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm(x))
        # x = x + self.mlp(self.layer_norm(x))
        return x

class EncoderLayer_ResNon(nn.Module):
    def __init__(self, dim, num_heads, d_hid, qkv_bias=False, qk_scale=None, drop=0.1):
        super().__init__()
        self.self_attn = Attention(dim, num_heads, qkv_bias, qk_scale, drop=drop)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp1 = Mlp(dim, d_hid, drop=drop)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp2 = Mlp(dim, d_hid, drop=drop)
        self.layer_norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.fc = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x1 = x + self.self_attn(self.layer_norm1(x))
        x1 = x1 + self.mlp1(self.layer_norm2(x1))
        x1 = f.softmax(self.fc(x1), dim=2)
        x2 = x + self.mlp2(self.layer_norm3(x))
        output = x + x2 * x1
        return output


if __name__ == "__main__":
    x = torch.randn([2, 3, 4])
    # net1 = PositionalEncoding(4)
    # y = net1(x)
    # net2 = Attention(4)
    # z = net2(y)
    # net3 = Mlp(4,16)
    # m = net3(z)
    # net4 = EncoderLayer(4, 2, 16)
    # n = net4(x)
    # net5 = EncoderLayers(4,4,2)
    # s = net5(n)
    # net6 = EncoderLayer_ResNon(4,2,16)
    # v = net6(x)
    net7 = EncoderLayer_mask(4, 2, 16)
    q = net7(x)



