
                       
                                     
                                       


import torch
from torch import Tensor, nn

def gram_schmidt(input):
    def projection(u, v):
        return (torch.dot(u.view(-1), v.view(-1)) / torch.dot(u.view(-1), u.view(-1))) * u
    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x / x.norm(p=2)
        output.append(x)
    return torch.stack(output)

def initialize_orthogonal_filters(c, h, w):
    if h * w < c:
        n = c // (h * w)
        gram = []
        for i in range(n):
            gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
        return torch.cat(gram, dim=0)
    else:
        return gram_schmidt(torch.rand([c, 1, h, w]))

class GramSchmidtTransform(torch.nn.Module):
    instance = {}
    constant_filter: Tensor

    @staticmethod
    def build(c: int, h: int):
        if (c, h) not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def __init__(self, c: int, h: int):
        super().__init__()
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h).view(c, h, h)
                            
        self.register_buffer("constant_filter", rand_ortho_filters.detach())

    def forward(self, x):
        _, _, h, w = x.shape
        _, H, W = self.constant_filter.shape
        if h != H or w != W: 
            x = torch.nn.functional.adaptive_avg_pool2d(x, (H, W))
                                   
        return (self.constant_filter.to(x.device) * x).sum(dim=(-1, -2), keepdim=True)


class Orthogonal_Channel_Attention(nn.Module):
    def __init__(self, channels: int, height: int = None):
        """
        初始化 Orthogonal_Channel_Attention 模块
        :param channels: 输入张量的通道数 (C)
        :param height: 可选，Gram-Schmidt 变换所需的高度 (H=W)，
                       如果不传则在第一次 forward 时自动使用输入的 H。
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.channels = channels
        self.height = height                           
        self.F_C_A = None         

                              
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 输入张量 (B, C, H, W)
        :return: 输出张量 (B, C, H, W)
        """
        B, C, H, W = x.shape

                                     
        if self.height is None:
            self.height = H
            self.F_C_A = GramSchmidtTransform.build(C, self.height)

                             
        if self.F_C_A is None:
            self.F_C_A = GramSchmidtTransform.build(C, self.height)

                         
        transformed = self.F_C_A(x)                

               
        compressed = transformed.view(B, C)
        excitation = self.channel_attention(compressed).view(B, C, 1, 1)

            
        output = x * excitation
                              
        return output

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = 128
    attention_module = Orthogonal_Channel_Attention(channels).to(device)  
    input_tensor = torch.rand(1, channels, 64, 64).to(device)
    output_tensor = attention_module(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")
