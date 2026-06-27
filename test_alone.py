from util.metric_tool import ConfuseMatrixMeter
import torch
from option import Options
from data.cd_dataset_edge import DataLoader
from model.create_model_edge import create_model
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import random

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    SEED = 3408
    setup_seed(SEED)
    print(f"Random seed fixed to {SEED}")

    opt = Options().parse()
    opt.phase = 'test'
    opt.batch_size = opt.test_batch_size

    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    print("#testing images = %d" % test_size)

    opt.load_pretrain = True
    model = create_model(opt)

    vis_dir = opt.vis_dir
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        print(f"Created visualization directory: {vis_dir}")

    tbar = tqdm(test_data, ncols=100)
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()

    model.eval()
    with torch.no_grad():
        for i, _data in enumerate(tbar):
            val_pred = model.inference(_data['img1'].cuda(), _data['img2'].cuda())
            val_target = _data['cd_label'].detach()
            val_pred = torch.argmax(val_pred.detach(), dim=1)

            _ = running_metric.update_cm(pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy())

            for j in range(val_pred.shape[0]):
                original_filename = os.path.basename(_data['fname'][j])

                pred_filename = original_filename

                pred_filepath = os.path.join(vis_dir, pred_filename)

                pred_mask = val_pred[j].cpu().numpy().astype(np.uint8)

                pred_mask[pred_mask == 1] = 255

                pred_image = Image.fromarray(pred_mask)
                pred_image.save(pred_filepath)

        print("\nLast batch image shape:", _data['img1'].shape, "Filenames:", _data['fname'])

        val_scores = running_metric.get_scores()
        message = '(phase: %s) ' % (opt.phase)
        for k, v in val_scores.items():
            message += '%s: %.3f ' % (k, v * 100)

        print(message)
