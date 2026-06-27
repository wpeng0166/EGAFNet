                           
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.block.wave import DWT_2D, IDWT_2D
from model.block.eucb import EUCB

class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel, 128, 1),
            norm_layer(128),
            act_layer(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            norm_layer(128),
            act_layer(),
            nn.Conv2d(128, channel, 1),
            norm_layer(channel)
        )

    def forward(self, x):
        return self.block(x)


class Scharr(nn.Module):
    def __init__(self, channel, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(Scharr, self).__init__()
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.conv_x = nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False)
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        self.norm = norm_layer(channel)
        self.act = act_layer()
        self.conv_extra = Conv_Extra(channel, norm_layer, act_layer)

    def forward(self, x):
        edge_x = self.conv_x(x)
        edge_y = self.conv_y(x)
                                                      
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        edge = self.act(self.norm(edge))
        return self.conv_extra(x * edge +x)


class TripleDilatedDWConv(nn.Module):
    """
    连续三个深度可分离空洞卷积，dilation=1,2,5
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

              
        if activation.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif activation.lower() == 'leakyrelu':
            act_layer = nn.LeakyReLU(0.2, inplace=True)
        elif activation.lower() == 'gelu':
            act_layer = nn.GELU()
        else:
            raise NotImplementedError(f"activation {activation} not implemented")

                                 
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=1, dilation=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            act_layer,
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            act_layer
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=2, dilation=2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            act_layer,
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            act_layer
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=5, dilation=5, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            act_layer,
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(out1))
        out2 = self.bn(out2)
        out2 += x                       
        return out2




class Fusion(nn.Module):
    def __init__(self, in_channels, wave):
        super(Fusion, self).__init__()
        self.dwt = DWT_2D(wave)
        self.convh1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.high = ResNet(in_channels)
        self.convh2 = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.convl = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.low = ResNet(in_channels)
        self.up = EUCB(128,128,3,1,'relu')
        self.sa = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False),         
            nn.Sigmoid()                  
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(in_channels , in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.proj1 = nn.Sequential(
            nn.Conv2d(in_channels *2 , in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.proj41 = nn.Sequential(
            nn.Conv2d(in_channels * 4 , in_channels * 2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True),
        )
        self.proj14 = nn.Sequential(
            nn.Conv2d(in_channels * 2  , in_channels *4, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels*4),
            nn.ReLU(inplace=True),
        )
        self.dconv = TripleDilatedDWConv(in_channels,in_channels,3,'relu')
        self.idwt = IDWT_2D(wave)
                                                     
        self.edge = Scharr(in_channels*2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
        )
                     
                                                                                   
                               
                               
                              
                                            
                                           
                                 
                                 
                                  
                                   
                                   

           
                     
                                                        


                                    
                               
                            

                                           
                                   

                         
                                                   
    def forward(self, x1,x2):
        b, c, h, w = x1.shape
        x_dwt = self.dwt(x1)
        ll, lh, hl, hh = x_dwt.split(c, 1)
                                           
                                 
                                 
                                  
        b1, c1, h1, w1 = ll.shape
                                        
        x2 = self.up(x2)
        map = self.sa(x2)
                                         
        b2, c2, h2, w2 = x2.shape
         
        if(h1!=h2):
            x2 =F.pad(x2, (0, 0, 1, 0), "constant", 0)
        low=torch.cat([ll, x2], 1)
        low = self.proj1(low)
        lowf=self.low(low)
        all = torch.cat([lowf,lh, hl, hh], 1)
        all_s = self.proj41(all)
                                     
        all_e = self.edge(all_s)
        all_l = self.proj14(all_e)
        all_lm = all_l * map + all_l
        out_idwt = self.idwt(all_lm)
        out_d = self.dconv(out_idwt)
        out = self.conv1(out_d)
        return out

                      
                               
                               
                              
                                            
                                             
                                   
                                   
                                    
                                   
                                          
                          
                             
                                           
                                   
                     
                                                        
                                    
                               
                            
                                               
                                    
                                       
                                    
                                      
                                        
                                   
                                      
                                 
                    

                     
                               
                               
                              
                                            
                                             
                                   
                                   
                                    
                                   
                                          
                          
                           
                                           
                                   
                     
                                                        
                                    
                               
                            
                                               
                                  
                                  
                                    
                                        
                                     
                                      
                                 
                    
                
                               
                               
                              
                                            
                                   
                          
                           
                                   
                     
                                                        
                                      
                                 
                              
                                             
                                  
                                  
                                    
                                        
                                     
                                      
                                 
                    
                  
                               
                               
                              
                                            
                                                                    
                                               
                                                                    
                                   
                                   
                                   
                                                                    
                                             
                                   
                                   
                                    
                                   
                                          
                          
                           
                                           
                                   
           
                     
                                                        
                                    
                               
                            
                                               
                                  
                                       
                                  
                                    
                                      
                                      
                                      
                                 
                    
                     
                               
                               
                                
                                              
                                                                                       
                 
                 
                 
                                             
                                   
                                   
                                    
                                   
                                          
                          
                           
                                           
                                   
           
                     
                                                        
                                    
                               
                            
                                               
                                  
                                       
                                  
                                    
                                      
                                                
                                                                                                    
                                      
                                 
                    

if __name__=="__main__":
    f1 = torch.randn(16,128,64,64)
    f4 = torch.randn(16,128,8,8)
                                                
    model = Fusion(128,wave='haar')
    y = model(f1,f4)
    print(y.shape)




                          
                                 
                                      
                  
                    
