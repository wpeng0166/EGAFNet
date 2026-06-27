import torch
import torch.nn as nn
from model.block.dsam import DSAM
from model.block.edge_new import Fusion
class EOL_Fusion_Block(nn.Module):
    def __init__(self, channels, k=8):
        super(EOL_Fusion_Block, self).__init__()
        self.channels = channels
        self.k = k
        if self.channels % self.k != 0:
            raise ValueError(f"input channels ({self.channels}) must be divisible by k ({self.k})")
        self.group_channels = self.channels // self.k
        interleaved_channels = self.channels + self.k
        self.gconv1 = nn.Conv2d(interleaved_channels, self.channels, kernel_size=3, padding=1, groups=1)
        self.gconv4 = nn.Conv2d(interleaved_channels, self.channels, kernel_size=3, padding=1, groups=4)
        self.gconv8 = nn.Conv2d(interleaved_channels, self.channels, kernel_size=3, padding=1, groups=8)
        self.bn = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature_map, edge_map):
        batch_size = feature_map.size(0)
        grouped_features = feature_map.view(batch_size, self.k, self.group_channels, *feature_map.shape[2:])
        interleaved_tensors = []
        for i in range(self.k):
            interleaved_tensors.append(grouped_features[:, i, ...])
            interleaved_tensors.append(edge_map)
        x_e = torch.cat(interleaved_tensors, dim=1)
        fused = self.gconv1(x_e) + self.gconv4(x_e) + self.gconv8(x_e)
        fused = self.relu(self.bn(fused))
        return feature_map + fused

class EARM(nn.Module):
    def __init__(self, channels=128):
        super(EARM, self).__init__()
        self.channels = channels
        self.edge_predictor = Fusion(channels,wave='haar')
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.conv_to_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.insert = EOL_Fusion_Block(self.channels, 8)
        self.cbam_p2 = DSAM(self.channels)
        self.cbam_p3 = DSAM(self.channels)
        self.cbam_p4 = DSAM(self.channels)
        self.cbam_p5 = DSAM(self.channels)

    def forward(self, features):
        diff_p2, diff_p3, diff_p4, diff_p5 = features
        raw_edge_pred = self.edge_predictor(diff_p2, diff_p5)
        edge_map_p2 = self.conv_to_attention(raw_edge_pred)

        edge_map_p3 = self.maxpool1(edge_map_p2)                  
        edge_map_p4 = self.maxpool2(edge_map_p3)                  
        edge_map_p5 = self.maxpool3(edge_map_p4)                
        guided_p2 = diff_p2 * edge_map_p2 + diff_p2
        guided_p3 = diff_p3 * edge_map_p3 + diff_p3
        guided_p4 = diff_p4 * edge_map_p4 + diff_p4
        guided_p5 = diff_p5 * edge_map_p5 + diff_p5

        refined_p2 = self.cbam_p2(guided_p2)
        refined_p3 = self.cbam_p3(guided_p3)
        refined_p4 = self.cbam_p4(guided_p4)
        refined_p5 = self.cbam_p5(guided_p5)
        return raw_edge_pred, (refined_p2, refined_p3, refined_p4, refined_p5)
