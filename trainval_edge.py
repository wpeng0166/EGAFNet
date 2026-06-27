import torch
from option import Options
from data.cd_dataset_edge import DataLoader
from model.create_model_edge import create_model
from tqdm import tqdm
import math
from util.metric_tool import ConfuseMatrixMeter
import os
import numpy as np
import random
from thop import profile

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

class Trainval(object):
    def __init__(self, opt):
        self.opt = opt

        train_loader = DataLoader(opt)
        self.train_data = train_loader.load_data()
        train_size = len(train_loader)
        print("#training images = %d" % train_size)
        opt.phase = 'test'
        val_loader = DataLoader(opt)
        self.val_data = val_loader.load_data()
        val_size = len(val_loader)
        print("#validation images = %d" % val_size)
        opt.phase = 'train'

        self.model = create_model(opt)
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular

        self.device = opt.gpu_ids
        print(self.device)

        self.iters = 0
        self.total_iters = math.ceil(train_size / opt.batch_size) * opt.num_epochs
        self.previous_best = 0.0
        self.running_metric = ConfuseMatrixMeter(n_class=2)

    def model_profile_mac_params(self):
        pre_img = torch.randn(1, 3, 256, 256).to(self.device[0])
        post_img = torch.randn(1, 3, 256, 256).to(self.device[0])
        label = torch.randint(0, 2, (1, 256, 256)).long().to(self.device[0])
        edge_label = torch.randint(0, 2, (1, 256, 256)).long().to(self.device[0])
        mac, prarms = profile(self.model, (pre_img, post_img, label, edge_label))
        print(f"FLOPs: {mac*2 / 1e9:.12f} G")
        print(f"Number of parameters: {prarms / 1e6:.12f} M")

    def train(self):
        tbar = tqdm(self.train_data, ncols=120)
        opt.phase = 'train'
        _loss = 0.0
        _focal_loss = 0.0
        _dice_loss = 0.0
        _edge_loss = 0.0
        for i, data in enumerate(tbar):

            self.model.detector.train()
            focal, dice, p2_loss, p3_loss, p4_loss, p5_loss, edge_loss = self.model(data['img1'].cuda(), data['img2'].cuda(), data['cd_label'].cuda(), data['edge_label'].cuda())
            loss = focal * 0.5 + dice + p3_loss + p4_loss + p5_loss +1.0 * edge_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.schedular.step()
            _loss += loss.item()
            _focal_loss += focal.item()
            _dice_loss += dice.item()
            _edge_loss += edge_loss.item()
            del loss

            tbar.set_description("Loss: %.3f, Focal: %.3f, Dice: %.3f, Edge: %.3f, LR: %.6f" %
                                 (_loss / (i + 1), _focal_loss / (i + 1), _dice_loss / (i + 1), _edge_loss / (i + 1), self.optimizer.param_groups[0]['lr']))

    def val(self):
        tbar = tqdm(self.val_data, ncols=120)
        self.running_metric.clear()
        opt.phase = 'test'
        self.model.eval()

        with torch.no_grad():
            for i, _data in enumerate(tbar):
                val_pred = self.model.inference(_data['img1'].cuda(), _data['img2'].cuda())
                val_target = _data['cd_label'].detach()
                val_pred = torch.argmax(val_pred.detach(), dim=1)
                _ = self.running_metric.update_cm(pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy())
            val_scores = self.running_metric.get_scores()
            message = '(phase: %s) ' % (self.opt.phase)
            for k, v in val_scores.items():
                message += '%s: %.3f ' % (k, v * 100)
            print(message)

        if val_scores['F1_1'] >= self.previous_best:
            self.model.save(self.opt.name, self.opt.backbone)
            self.previous_best = val_scores['F1_1']

if __name__ == "__main__":
    setup_seed(seed=63)
    opt = Options().parse()
    trainval = Trainval(opt)
    for epoch in range(1, opt.num_epochs + 1):
        print("\n==> Name %s, Epoch %i, previous best F1_1 = %.3f" % (opt.name, epoch, trainval.previous_best * 100))
        trainval.train()
        trainval.val()
