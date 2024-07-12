from model.Unet import *
import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F
from einops import rearrange

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class spa_spec_transformer(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, (dim_head * heads) , bias=False)
        self.temperature = nn.Parameter(torch.ones(1,self.num_heads,1,1))
        self.q = nn.Conv2d(dim,dim*4,kernel_size=1,bias=False)
        self.k = nn.Conv2d(dim,dim*4,kernel_size=1,bias=False)
        self.q_conv = nn.Conv2d(dim*4,dim*4,kernel_size=3,padding=1,groups=dim,bias=False)
        self.k_conv = nn.Conv2d(dim*4, dim*4, kernel_size=3, padding=1, groups=dim, bias=False)

        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale2 = nn.Parameter(torch.ones(heads,1,1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
        nn.Conv2d(dim_head * heads, dim, 3, 1, 1, bias=False, groups=dim),
        GELU(),
        nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, mask=None):

        b2, c2, h2, w2 = x_in.shape
        x_in2 = x_in.permute(0,2,3,1)
        b, h, w, c = x_in2.shape

        x_spectral = x_in2.reshape(b,h*w,c)

        q_inp = self.to_q(x_spectral)
        k_inp = self.to_k(x_spectral)
        v_inp = self.to_v(x_spectral)

        q2 = self.q_conv(self.q(x_in))
        k2 = self.k_conv(self.k(x_in))
        q2 = q2.reshape(b, self.num_heads, -1, h2 * w2)
        k2 = k2.reshape(b, self.num_heads, -1, h2 * w2)
        q2, k2 = F.normalize(q2, dim=-1), F.normalize(k2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
        (q_inp, k_inp, v_inp))

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        attn = (q @ k.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)

        attn2 = (q2 @ k2.transpose(-2,-1))
        attn2 = attn2 * self.rescale2
        attn2 = attn2.softmax(dim=-1)

        attn3 = attn * attn2
        attn3 = attn3.softmax(dim=-1)
        x = attn3 @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)

        out_c = self.proj(x).view(b, h, w, c)

        out_p = self.pos_emb(v_inp.reshape(b,h,w,4*c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p
        out = out.permute(0, 3, 1, 2)

        return out

class MSCNN(nn.Module):
    def __init__(self, ch):
        super(MSCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.projection = nn.Conv2d(in_channels=96, out_channels=ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out = torch.cat([
            out1, out2, out3
        ], dim=1)

        out_finnal = self.projection(out)

        return out_finnal

class MS_CC_Net(nn.Module):
    def __init__(self, RGB_ch, MS_ch):
        super(MS_CC_Net, self).__init__()

        self.Local_illum_en = UNet_encoder(in_ch=RGB_ch)
        self.local_illum_de = UNet_for_CC_decoder(out_ch=3)
        self.conf_en = UNet_encoder(in_ch=MS_ch)
        self.conf_de = UNet_for_CC_decoder_conf(out_ch=1)
        self.mscnn = MSCNN(ch=RGB_ch)

        self.attention = spa_spec_transformer(dim=MS_ch, dim_head= MS_ch*2, heads=2)
        self.illum_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.illum_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)

    def forward(self, RGB, MS):
        ''' spectral '''
        spectral_original = MS

        ''' spatial'''
        spatial = RGB

        ''' confidence network '''
        spectral_feature = self.attention(spectral_original)
        spectral_feature = spectral_feature + spectral_original
        # confidence map
        conf1, conf2, conf3, conf4, conf5 = self.conf_en(spectral_feature)
        conf = self.conf_de(conf5, conf4, conf3)
        conf_map_re = torch.reshape(conf, [conf.size(0), conf.size(2) * conf.size(3)])
        sum = torch.sum(conf_map_re, dim=1)
        sum = sum.reshape([sum.size(0), 1, 1, 1])
        conf = conf / (sum.repeat([1, 1, conf.size(2), conf.size(3)]) + 0.00001)
        conf_map = conf.repeat([1, 3, 1, 1])

        ''' local illuminant network '''
        # spatial
        spatial_feature = self.mscnn(spatial) * spatial
        L1, L2, L3, L4, L5 = self.Local_illum_en(spatial_feature)
        # local illuminant map
        local_illum = self.local_illum_de(L5, L4, L3)
        normal = (torch.norm(local_illum[:, :, :, :], p=2, dim=1, keepdim=True) + 1e-04)
        local_illum = local_illum[:, :, :, :] / normal

        ''' illuminant output '''
        local_il = local_illum * conf_map
        local_il = local_il.reshape(local_il.size(0), local_il.size(1), local_il.size(2) * local_il.size(3))
        weighted_sum = torch.sum(local_il, dim=2)
        pred = weighted_sum / (torch.norm(weighted_sum, dim=1, p=2, keepdim=True) + 0.000001)

        '''contrastive loss'''
        local_illum2 = self.illum_conv2(self.illum_conv1(local_illum))
        illum = normalize(torch.sum(torch.sum(local_illum2, 2), 2), dim=1)

        '''spectral feature'''
        spectral_feature = torch.mean(spectral_feature,dim=1)

        return pred, local_illum, conf_map, illum
