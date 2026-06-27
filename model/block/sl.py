import torch
import torch.nn as nn

                                
class DepthwiseSeparableConv(nn.Module):
    """
    一个基础的深度可分离卷积块。
    包含: Depthwise Conv -> BN -> ReLU -> Pointwise Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_part = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size,
                stride=stride, padding=padding, groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise_part = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.depthwise_part(x)
        x = self.pointwise_part(x)
        return x

class ResidualMultiBranchModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualMultiBranchModule, self).__init__()
        
        self.inter_channels = 128

                  
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=True)
        )

                           
        self.branch_k3 = DepthwiseSeparableConv(self.inter_channels, self.inter_channels, 3, padding=1)
        self.branch_k5 = DepthwiseSeparableConv(self.inter_channels, self.inter_channels, 5, padding=2)
        self.branch_k7 = DepthwiseSeparableConv(self.inter_channels, self.inter_channels, 7, padding=3)

                                         
        self.final_conv = nn.Conv2d(self.inter_channels, out_channels, kernel_size=1, bias=False)



                            
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                               
        x_processed = self.initial_block(x)
        
                  
        b3_out = self.branch_k3(x_processed)
        b5_out = self.branch_k5(x_processed)
        b7_out = self.branch_k7(x_processed)
        x_fused = b3_out + b5_out + b7_out
        
                   
        x_main_path = self.final_conv(x_fused)
        
                    
                                                           
        x_added = x_main_path + x_processed
        
                          
        output = self.final_bn(x_added)
        output = self.final_relu(output)
        
        return output

              
if __name__ == '__main__':
    x = torch.randn(16,512,8,8)
    mod = ResidualMultiBranchModule(512,128)
    y = mod(x)
    print(y.shape)