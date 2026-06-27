import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
                                                                                                                              

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

              
class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a1 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 4, 7, padding=3, groups=dim // 4)
        )
        self.v1 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v11 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v12 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.conv3_1 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim//4)

        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.a2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, 9, padding=4, groups=dim // 2)
        )
        self.v2 = nn.Conv2d(dim//2, dim//2, 1)
        self.v21 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v22 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj2 = nn.Conv2d(dim // 2, dim // 4, 1)
        self.conv3_2 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.norm3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")
        self.a3 = nn.Sequential(
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 11, padding=5, groups=dim * 3 // 4)
        )
        self.v3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v31 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v32 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj3 = nn.Conv2d(dim * 3 // 4, dim // 4, 1)
        self.conv3_3 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.dim = dim

    def forward(self, x):

        x = self.norm1(x)
        x_split = torch.split(x, self.dim // 4, dim=1)
        a = self.a1(x_split[0])
        mul = a * self.v1(x_split[0])
        mul = self.v11(mul)
        x1 = self.conv3_1(self.v12(x_split[1]))
        x1 = x1 + a
        x1 = torch.cat((x1, mul), dim=1)

        x1 = self.norm2(x1)
        a = self.a2(x1)
        mul = a * self.v2(x1)
        mul = self.v21(mul)
        x2 = self.conv3_2(self.v22(x_split[2]))
        x2 = x2 + self.proj2(a)
        x2 = torch.cat((x2, mul), dim=1)

        x2 = self.norm3(x2)
        a = self.a3(x2)
        mul = a * self.v3(x2)
        mul = self.v31(mul)
        x3 = self.conv3_3(self.v32(x_split[3]))
        x3 = x3 + self.proj3(a)
        x = torch.cat((x3, mul), dim=1)

        return x
                  
                           
                              
                            

                                                                             
                                  
                                               
                        
                                                             
                                                                          
           
                                                    
                                                     
                                                     
                                                                                   

                                                                                  
                                  
                                               
                        
                                                             
                                                                          
           
                                                
                                                     
                                                     
                                                       
                                                                                     

                                                                                      
                                  
                                                       
                        
                                                              
                                                                                      
           
                                                            
                                                             
                                                     
                                                           
                                                                                     

                        

                           

                           
                                                        
                                 
                                       
                             
                                                 
                     
                                          

                             
                         
                               
                             
                                                 
                                 
                                          

                             
                         
                               
                             
                                                 
                                 
                                             

                      



if __name__ =="__main__":
    model = ConvMod(128)
    x = torch.randn(16,128,64,64)
    y = model(x)
    print(y.shape)

