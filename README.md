# EGAFNet: Edge-Guided Adaptive Fusion Network With Spatial-Frequency Interaction for Remote Sensing Change Detection

![Python](https://img.shields.io/badge/python-3.8-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c.svg) ![CUDA](https://img.shields.io/badge/CUDA-11.8-76b900.svg) ![License](https://img.shields.io/badge/license-Academic-4c1.svg)

> **Authors:** Peng Wang, Yangbin Liu, Quanqing Ma, Qingzhan Zhao, Zhiming Fang, Yuchen Zheng  
> **Journal:** IEEE Transactions on Geoscience and Remote Sensing (TGRS)  
> **Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11506587)  
> **DOI:** [10.1109/TGRS.2026.3690504](https://doi.org/10.1109/TGRS.2026.3690504)

If you find this paper or code helpful, please consider giving this repository a star.

## News

- The code and trained model weights are released for remote sensing binary change detection.
- Trained checkpoints are provided for SYSU, SECOND, and JLYHCD.

## Introduction

EGAFNet is an edge-guided adaptive fusion network for remote sensing change detection. The network combines edge-aware guidance with spatial-frequency interaction to enhance multilevel feature fusion and improve the discrimination of changed regions.

## Environment

The released code has been checked in the following environment:

```text
Python       3.8
PyTorch      2.1.0
TorchVision  0.16.0
CUDA         11.8
NumPy        1.24.4
Pillow       10.4.0
timm         1.0.15
einops       0.8.1
Kornia       0.7.3
THOP         2.0.15
OpenCV       4.6.0
PyWavelets   1.4.1
```

A typical installation is:

```bash
conda create -n egafnet python=3.8 -y
conda activate egafnet

# Install PyTorch according to your CUDA version.
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

pip install numpy==1.24.4 pillow==10.4.0 tqdm==4.65.2 timm==1.0.15 einops==0.8.1
pip install kornia==0.7.3 thop==2.0.15 opencv-python==4.6.0.66 PyWavelets==1.4.1
```

If your CUDA version is different, please install the matching PyTorch build from the official PyTorch instructions and keep the remaining packages consistent.

## Dataset Preparation

Organize each dataset as follows:

```text
DATA_ROOT/
`-- DATASET_NAME/
    |-- train/
    |   |-- A/
    |   |-- B/
    |   |-- label/
    |   `-- edge/
    |-- val/
    |   |-- A/
    |   |-- B/
    |   |-- label/
    |   `-- edge/
    `-- test/
        |-- A/
        |-- B/
        |-- label/
        `-- edge/
```

Where:

- `A/`: images at time T1
- `B/`: images at time T2
- `label/`: binary change masks
- `edge/`: binary edge labels

The edge labels can be generated from the original change masks using `edge_make.py`.

By default, the code expects datasets under:

```text
/home/s_wp/workspace/data/
```

You can change the path with `--dataroot`.

## Training

Train on SYSU:

```bash
python trainval_edge.py \
  --name SYSU \
  --dataset SYSU \
  --dataroot /home/s_wp/workspace/data/ \
  --input_size 256
```

Train on SECOND:

```bash
python trainval_edge.py \
  --name SECOND \
  --dataset SECOND \
  --dataroot /home/s_wp/workspace/data/ \
  --input_size 512
```

Train on JLYHCD:

```bash
python trainval_edge.py \
  --name JLYHCD \
  --dataset JLYHCD \
  --dataroot /home/s_wp/workspace/data/ \
  --input_size 256
```

You may adjust `--input_size`, `--batch_size`, `--num_epochs`, and `--lr` according to your GPU memory and dataset setting.

## Trained Weights

The trained model weights are not included in this repository. Please download the `.pth` files from the links below and place them under `checkpoints/`:

```text
checkpoints/SYSU/SYSU_mobilenetv2_best.pth
checkpoints/SECOND/SECOND_mobilenetv2_best.pth
checkpoints/JLYHCD/JLYHCD_mobilenetv2_best.pth
```

| Weight file | Baidu Netdisk | Google Drive |
| --- | --- | --- |
| `SYSU_mobilenetv2_best.pth` | [Download](https://pan.baidu.com/s/11oasOEicZODDJF-xCvGcJg?pwd=wpfx), code: `wpfx` | [Download](https://drive.google.com/file/d/1LNsmcvFpBGiCinKZbDenl-xhMjezzgK5/view?usp=sharing) |
| `SECOND_mobilenetv2_best.pth` | [Download](https://pan.baidu.com/s/1uSJ53qi-d5tzS4ilTqBAyA?pwd=wpfx), code: `wpfx` | [Download](https://drive.google.com/file/d/1yf_RlxRQsHkzMLdWqTDrpr5YFFH2p_aL/view?usp=sharing) |
| `JLYHCD_mobilenetv2_best.pth` | [Download](https://pan.baidu.com/s/15qzsWtf2UcudUEPJu1a3Bw?pwd=wpfx), code: `wpfx` | [Download](https://drive.google.com/file/d/173JsgHWudws9mRMt43eJ3EK_RXPCKxYB/view?usp=sharing) |

## Testing

Run inference on SYSU:

```bash
python test_alone.py \
  --name SYSU \
  --dataset SYSU \
  --dataroot /home/s_wp/workspace/data/ \
  --vis_dir checkpoints/SYSU/results
```

Run inference on SECOND:

```bash
python test_alone.py \
  --name SECOND \
  --dataset SECOND \
  --dataroot /home/s_wp/workspace/data/ \
  --vis_dir checkpoints/SECOND/results
```

Run inference on JLYHCD:

```bash
python test_alone.py \
  --name JLYHCD \
  --dataset JLYHCD \
  --dataroot /home/s_wp/workspace/data/ \
  --vis_dir checkpoints/JLYHCD/results
```

Predicted binary masks will be saved to the corresponding `--vis_dir`.

## Citation

If this work is useful for your research, please cite:

```bibtex
@ARTICLE{11506587,
  author={Wang, Peng and Liu, Yangbin and Ma, Quanqing and Zhao, Qingzhan and Fang, Zhiming and Zheng, Yuchen},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={EGAFNet: Edge-Guided Adaptive Fusion Network With Spatial-Frequency Interaction for Remote Sensing Change Detection},
  year={2026},
  volume={64},
  pages={5620817-5620817},
  doi={10.1109/TGRS.2026.3690504}
}
```

## License

This repository is released for academic research. Please follow the license terms of the datasets and third-party libraries used in this project.

## Contact

For questions about the paper or code, please open an issue in this repository.
