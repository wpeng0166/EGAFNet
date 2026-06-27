import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
              
                                                                                        

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class DepthwiseSeparableConv(nn.Module):
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

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
    


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)




class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8
                                             
    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output
                          
                           
                                    
                                                           
                                                             

                                             
                                                                                                              
                                                                                                              
                                                                                                              

                                          
                                                                         
                                               
                                                                             
                                                                             
                                                                             

                               
                                                                 
                                     

                          
                                           
        
                        
                                                                           
                                                                                                   

                                            
                                                           

                       
                     
                           
                                    
                                                      
                                                             
                              

                         
                                                      
                                      
                               
        
                                  
                                                                                   
                                                          
                                                          

                                   
                                   
                                   

                                               
                                                                                
                                                                   
                                     

                          

                   
                                                                                   
                                                                            

                              
                                        
                       
                                   
class ResidualFSASBlock(nn.Module):
    """
    一个封装了 FSAS 模块并带有残差连接的完整处理块。
    流程: Pre-process -> (FSAS -> Post-Conv) + Shortcut -> BN -> ReLU
    """
    def __init__(self, in_dim, embed_dim, out_dim=None, bias=False):
        """
        初始化带残差的 FSAS 块。
        
        参数:
            in_dim (int): 输入特征的通道数。
            embed_dim (int): FSAS 模块内部处理的通道数。
            out_dim (int, optional): 输出特征的通道数。如果为 None，则默认为 in_dim。
            bias (bool): 卷积层是否使用偏置。
        """
        super(ResidualFSASBlock, self).__init__()
        
        if out_dim is None:
            out_dim = in_dim

               
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_dim, embed_dim, kernel_size=1, bias=bias),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

                    
        self.fsas_core = FSAS(dim=embed_dim, bias=bias)

                      
                           
        self.post_conv = nn.Conv2d(embed_dim, out_dim, kernel_size=1, bias=bias)
                       
        self.post_bn = nn.BatchNorm2d(out_dim)
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
                          
        x_preprocessed = self.preprocess(x)
        
                     
                         
        x_main = self.fsas_core(x_preprocessed)
                         
        x_main = self.post_conv(x_main)
                                  
        x_added = x_main + x_preprocessed
        
                            
        output = self.post_bn(x_added)
        output = self.post_relu(output)
        
        return output
                     
                           
                            
                                             
        
                       
                           
                                                 
                           
                                         
                                    
                           
        
                              
                                        
                                         
                                          
                       




                                                    
                                     
                                                                      
                                                   
        
                             
                              
                                                                                    
                 
                                          
                                                                     
                                        
                                   
           
                                         
                                                         
                                                                                  
                                                
                                                

                                                        
        
                           
                                             
                                                      
                                                 
                                         
                                           
                                        
                                         
                       
                                                                   
                                     
                                                                      
                                                   
                             
                              
                                                                                
                                          
                                                                     
                                        
                                   
           
                                           
                                                         
                                                                                  
                                                
                                                

                                                          
        
                           
                                             
                                              
                                                 
                                         
                                                 
                                                          
                                        
                                               
                       
                                              
                                     
                                                                      
                                                   
                             
                              
                                                                                  
                                          
                                                                     
                                        
                                   
           
                                      
                                                                                    
                                        
                                    
                                                              
           
                                           
                                                         
                                                                                  
                                                
                                                

                                                          
        
                           
                                  
                                           
                                                 
                                         
                                                 
                                
                                        
                                               
                       
if __name__ == '__main__':
    x = torch.randn(16,64,32,32)
    model = ResidualFSASBlock(64,128,128,bias=True)
    y = model(x)
    print(y.shape)
