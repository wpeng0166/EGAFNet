from .network_edge import Detectorv3
import torch
from torch import nn
import os
import torch.optim as optim
from .block.schedular import get_cosine_schedule_with_warmup
from .loss.focal import FocalLoss
from .loss.dice import DICELoss

def get_model(backbone_name='mobilenetv2', fpn_name='fpn', fpn_channels=128, deform_groups=4,
              gamma_mode='SE', beta_mode='contextgatedconv', num_heads=1, num_points=8, kernel_layers=1,
              dropout_rate=0.1, init_type='kaiming_normal', lge=False, fdconv= False ,arconv=False):
    detector = Detectorv3(backbone_name, fpn_name, fpn_channels, deform_groups, gamma_mode, beta_mode,
                        num_heads, num_points, kernel_layers, dropout_rate, init_type, lge=lge,fdconv=fdconv,arconv=arconv)

    log_file = './model.txt'
    with open(log_file, 'w') as f:
        print(detector, file=f)

    return detector
class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.device = torch.device("cuda:%s" % opt.gpu_ids[0] if torch.cuda.is_available() else "cpu")
        self.opt = opt
        self.base_lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.detector = get_model(backbone_name=opt.backbone, fpn_name=opt.fpn, fpn_channels=opt.fpn_channels,
                                  deform_groups=opt.deform_groups, gamma_mode=opt.gamma_mode, beta_mode=opt.beta_mode,
                                  num_heads=opt.num_heads, num_points=opt.num_points, kernel_layers=opt.kernel_layers,
                                  dropout_rate=opt.dropout_rate, init_type=opt.init_type, lge=opt.lge, fdconv=opt.fdconv)

        self.focal = FocalLoss(alpha=opt.alpha, gamma=opt.gamma)
        self.dice = DICELoss()

        self.optimizer = optim.AdamW(self.detector.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.schedular = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=625 * opt.warmup_epochs,
                                                         num_training_steps=625 * opt.num_epochs)
        if opt.load_pretrain:
            self.load_ckpt(self.detector, self.optimizer, opt.name, opt.backbone)
        self.detector.cuda()

        print("---------- Networks initialized -------------")

    def forward(self, x1, x2, label, edge_label):                             
        pred, pred_p2, pred_p3, pred_p4, pred_p5, edge_pred = self.detector(x1, x2)                                       
        label = label.long()
        focal = self.focal(pred, label)
        dice = self.dice(pred, label)
        p2_loss = self.focal(pred_p2, label) * 0.5 + self.dice(pred_p2, label)
        p3_loss = self.focal(pred_p3, label) * 0.5 + self.dice(pred_p3, label)
        p4_loss = self.focal(pred_p4, label) * 0.5 + self.dice(pred_p4, label)
        p5_loss = self.focal(pred_p5, label) * 0.5 + self.dice(pred_p5, label)
        edge_label = edge_label.long()
        edge_loss = self.focal(edge_pred, edge_label) + self.dice(edge_pred, edge_label)
        return focal, dice, p2_loss, p3_loss, p4_loss, p5_loss, edge_loss

    def inference(self, x1, x2):
        with torch.no_grad():
            pred, _, _, _, _, _ = self.detector(x1, x2)
        return pred

    def inference_cam(self, x1, x2):
        pred, _, _, _, _, _ = self.detector(x1, x2)
        return pred

    def load_ckpt(self, network, optimizer, name, backbone):
        save_filename = '%s_%s_best.pth' % (name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print(f"{save_path} not exists yet!")
            raise FileNotFoundError(f"Checkpoint file '{save_filename}' must exist!")
        else:
            checkpoint = torch.load(save_path, map_location=self.device)
            network.load_state_dict(checkpoint['network'], strict=False)
            print(f"✅ Model weights loaded from {save_filename}")

            if optimizer is not None and 'optimizer' in checkpoint:
                ckpt_opt = checkpoint['optimizer']
                cur_opt = optimizer.state_dict()

                print("\n========== 🔍 Optimizer structure check ==========")
                print(f"Checkpoint param group count: {len(ckpt_opt['param_groups'])}")
                print(f"Current optimizer param group count: {len(cur_opt['param_groups'])}")

                for i, (ckpt_g, cur_g) in enumerate(zip(ckpt_opt['param_groups'], cur_opt['param_groups'])):
                    ckpt_ids = set(ckpt_g['params'])
                    cur_ids = set(cur_g['params'])
                    print(f"\n--- Group {i} ---")
                    print(f"Checkpoint group param count: {len(ckpt_ids)}")
                    print(f"Current optimizer group param count: {len(cur_ids)}")
                    missing = ckpt_ids - cur_ids
                    new = cur_ids - ckpt_ids
                    if missing:
                        print(f"⚠️ Missing {len(missing)} params from current model.")
                    if new:
                        print(f"⚠️ New {len(new)} params not in checkpoint.")
                    if not missing and not new:
                        print("✅ Group matches.")

                print("==================================================\n")

                try:
                    optimizer.load_state_dict(ckpt_opt)
                    print("✅ Optimizer state loaded successfully.")
                except Exception as e:
                    print("Optimizer state skipped:", str(e))

    def save_ckpt(self, network, optimizer, model_name, backbone):
        save_filename = '%s_%s_best.pth' % (model_name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save({'network': network.cpu().state_dict(),
                    'optimizer': optimizer.state_dict()},
                   save_path)
        if torch.cuda.is_available():
            network.cuda()

    def save(self, model_name, backbone):
        self.save_ckpt(self.detector, self.optimizer, model_name, backbone)

    def name(self):
        return self.opt.name

def create_model(opt):
    model = Model(opt)
    print("model [%s] was created" % model.name())

    return model.cuda()
