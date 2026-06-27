import torch
import torch.nn as nn
from model.block.bigconv import ConvMod
from model.block.zjtd import Orthogonal_Channel_Attention
from model.block.spam import DiSpAM
class FUSION0(nn.Module):
    def __init__(self,dim ):
        super().__init__()
        self.bigconv = ConvMod(dim=dim)
        self.ca = Orthogonal_Channel_Attention(channels=dim)
        self.sa = DiSpAM(c=dim, DW_Expand=2, dilations=[1, 4, 9], extra_depth_wise=True)
        self.proj1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.bn = nn.BatchNorm2d(dim)
        self.gate1 = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gate2 = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gate_context = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gate_change = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self,x1,x2):
        x_sub = torch.abs(x1-x2)
        x_s = self.sa(x_sub)
        x11 = x1 * x_s
        x22 = x2 * x_s
        x11 = x11 + x1
        x11 = self.proj1(x11)
        x22 = x22 + x2
        x22 = self.proj1(x22)

        x_cat = torch.cat([x11,x22],dim=1)
        x_c = self.proj2(x_cat)
        x_c = self.ca(x_c)
        x111 = x11 * x_c
        x222 = x22 * x_c
        x_cat_f = torch.cat([x111,x222],dim=1)
        c_f = self.proj3(x_cat_f)
        out1 = self.bigconv(c_f)
        out = out1 + c_f
        return out

if __name__ == "__main__":
    x1 = torch.randn(3, 128, 64, 64)
    x2 = torch.randn(3, 128, 64, 64)
    cross_mala = FUSION0(dim=128)
    y1 = cross_mala(x1, x2)
    print(y1.shape)                    
