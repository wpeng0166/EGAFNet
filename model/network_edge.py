import torch.nn as nn
import torch.nn.functional as F
from .backbone.mobilenetv2 import mobilenet_v2
from .block.modulation import VerticalFusion
from .block.heads import GatedResidualUpHead
from .block.edge import EARM
from .block.cmala import FUSION0
from .block.fsas import ResidualFSASBlock
from .block.sl import ResidualMultiBranchModule

def get_backbone(backbone_name):
    if backbone_name == 'mobilenetv2':
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    else:
        raise NotImplementedError(f"BACKBONE [{backbone_name}] is not implemented!")
    return backbone

class Detectorv3(nn.Module):
    def __init__(self, backbone_name='mobilenetv2', fpn_name='fpn', fpn_channels=128,
                 deform_groups=4, gamma_mode='SE', beta_mode='contextgatedconv',
                 num_heads=1, num_points=8, kernel_layers=1, dropout_rate=0.1, init_type='kaiming_normal', lge=False, fdconv=False, arconv=False):
        super().__init__()
        self.backbone = get_backbone(backbone_name)

        self.earm = EARM(channels=fpn_channels)

        self.p5_to_p4 = VerticalFusion(fpn_channels, focal_window=5, focal_level=3)
        self.p4_to_p3 = VerticalFusion(fpn_channels, focal_window=3, focal_level=3)
        self.p3_to_p2 = VerticalFusion(fpn_channels, focal_window=1, focal_level=3)
        self.xijie1 = ResidualFSASBlock(24,128,128,True)
        self.xijie2 = ResidualFSASBlock(32,128,128,True)
        self.yuyi3 = ResidualMultiBranchModule(96,128)
        self.yuyi4 = ResidualMultiBranchModule(320,128)

        self.diff_mdfm_p2 = FUSION0(128)
        self.diff_mdfm_p3 = FUSION0(128)
        self.diff_mdfm_p4 = FUSION0(128)
        self.diff_mdfm_p5 = FUSION0(128)
        self.p5_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p4_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p3_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p2_head = nn.Conv2d(fpn_channels, 2, 1)
        self.project = nn.Sequential(
            nn.Conv2d(fpn_channels * 4, fpn_channels, 1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(True)
        )
        self.head = GatedResidualUpHead(fpn_channels, 2, dropout_rate=dropout_rate)
        self.visualize = False
        self.visualization_features = {}

    def forward(self, x1, x2):
        H, W = x1.shape[-2:]
        t1_c1, t1_c2, t1_c3, t1_c4, t1_c5 = self.backbone.forward(x1)
        t2_c1, t2_c2, t2_c3, t2_c4, t2_c5 = self.backbone.forward(x2)
        if self.visualize:
            self.visualization_features['t1_c2'] = t1_c2
            self.visualization_features['t1_c3'] = t1_c3
            self.visualization_features['t1_c4'] = t1_c4
            self.visualization_features['t1_c5'] = t1_c5
            self.visualization_features['t2_c2'] = t2_c2
            self.visualization_features['t2_c3'] = t2_c3
            self.visualization_features['t2_c4'] = t2_c4
            self.visualization_features['t2_c5'] = t2_c5

        t1_p2 = self.xijie1(t1_c2)
        t1_p3 = self.xijie2(t1_c3)
        t1_p4 = self.yuyi3(t1_c4)
        t1_p5 = self.yuyi4(t1_c5)
        t2_p2 = self.xijie1(t2_c2)
        t2_p3 = self.xijie2(t2_c3)
        t2_p4 = self.yuyi3(t2_c4)
        t2_p5 = self.yuyi4(t2_c5)

        if self.visualize:
            self.visualization_features['t1_p2'] = t1_p2
            self.visualization_features['t1_p3'] = t1_p3
            self.visualization_features['t1_p4'] = t1_p4
            self.visualization_features['t1_p5'] = t1_p5
            self.visualization_features['t2_p2'] = t2_p2
            self.visualization_features['t2_p3'] = t2_p3
            self.visualization_features['t2_p4'] = t2_p4
            self.visualization_features['t2_p5'] = t2_p5

        diff_p2 = self.diff_mdfm_p2(t1_p2, t2_p2)
        diff_p3 = self.diff_mdfm_p3(t1_p3, t2_p3)
        diff_p4 = self.diff_mdfm_p4(t1_p4, t2_p4)
        diff_p5 = self.diff_mdfm_p5(t1_p5, t2_p5)

        if self.visualize:
            self.visualization_features['diff_p2'] = diff_p2
            self.visualization_features['diff_p3'] = diff_p3
            self.visualization_features['diff_p4'] = diff_p4
            self.visualization_features['diff_p5'] = diff_p5

        edge_pred, (diff_p2, diff_p3, diff_p4, diff_p5) = self.earm((diff_p2, diff_p3, diff_p4, diff_p5))
        if self.visualize:
            self.visualization_features['diff_p2_e'] = diff_p2
            self.visualization_features['diff_p3_e'] = diff_p3
            self.visualization_features['diff_p4_e'] = diff_p4
            self.visualization_features['diff_p5_e'] = diff_p5

        fea_p5 = diff_p5
        pred_p5 = self.p5_head(fea_p5)
        fea_p4 = self.p5_to_p4(fea_p5, diff_p4)
        pred_p4 = self.p4_head(fea_p4)
        fea_p3 = self.p4_to_p3(fea_p4, diff_p3)
        pred_p3 = self.p3_head(fea_p3)
        fea_p2 = self.p3_to_p2(fea_p3, diff_p2)
        pred_p2 = self.p2_head(fea_p2)
        pred = self.head(fea_p2)

        target_size = (H, W)
        pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
        pred_p2 = F.interpolate(pred_p2, size=target_size, mode='bilinear', align_corners=False)
        pred_p3 = F.interpolate(pred_p3, size=target_size, mode='bilinear', align_corners=False)
        pred_p4 = F.interpolate(pred_p4, size=target_size, mode='bilinear', align_corners=False)
        pred_p5 = F.interpolate(pred_p5, size=target_size, mode='bilinear', align_corners=False)
        edge_pred = F.interpolate(edge_pred, size=target_size, mode='bilinear', align_corners=False)

        return pred, pred_p2, pred_p3, pred_p4, pred_p5, edge_pred
