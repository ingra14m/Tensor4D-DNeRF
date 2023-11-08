# Tensor4D : Efficient Neural 4D Decomposition for High-fidelity Dynamic Reconstruction and Rendering
### [Project Page](https://liuyebin.com/tensor4d/tensor4d.html) | [Paper](https://arxiv.org/abs/2211.11610) | [Data](https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view)

> [Tensor4D : Efficient Neural 4D Decomposition for High-fidelity Dynamic Reconstruction and Rendering](https://arxiv.org/pdf/2211.11610.pdf)  
> Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu, Hongwen Zhang, Yebin Liu.
> (**CVPR 2023 Highlight**)

This is an unofficial implementation of Tensor4D with support for the D-NeRF dataset. However, I cannot reproduce the impeccable results in the D-NeRF dataset as presented in the [Tensor4D paper](https://arxiv.org/abs/2211.11610). To preclude any suspicion that my own paper intentionally downplays the performance metrics of Tensor4D, I have made this repository public. Everyone is able to review my **commit history** to ascertain that I have not altered any of the core functions of Tensor4D. I think there maybe some bugs in my implementation, since Tensor4D is the CVPR 2023 highlight paper.

## Installation

To deploy and run Tensor4d, you will need install the following Python libraries

```
numpy
opencv-python
torch
tensorboard
shutil
tqdm
pyhocon==0.3.57
glob
scipy
einops
```

## Run the code on D-NeRF dataset

> You need to change `data_dir` in the config file to read D-NeRF Dataset

```shell
# train jumpingjacks
python exp_runner.py --case d-nerf --mode train --conf confs/D-NeRF/t4d_jump.conf --gpu 0
# render & metrics for jumpingjacks
python exp_runner.py --mode render_all --conf confs/t4d_jump.conf --gpu 0 --is_continue
```

- For rendering and assessing metrics, the default measurement pertains to the **training set**. To evaluate the test set, you need to modify the [splits](https://github.com/ingra14m/Tensor4D-DNeRF/blob/a71a008bc1e867c29b1f8f267ea1cbb17a665b79/models/dataset.py#L235) by changing `train` to `test`.

### Run the standard Tensor4D 

To train Tensor4D for monocular cases, you can use the following scripts:

```
# Train Tensor4D with flow
python exp_runner.py --case lego_v1 --mode train --conf confs/t4d_lego.conf --gpu 0    
# Resume training
python exp_runner.py --case lego_v1 --mode train --conf confs/t4d_lego.conf --gpu 0 --is_continue
```

After training, you can visualize the results by the following scripts:

```
# interpolation between view 0 and view 2, setting the number of interpolation views to 100 and the downsampling resolution to 2

python exp_runner.py --case t4d_lego --mode interpolate_0_2 --conf confs/t4d_lego.conf --is_continue --inter_reso_level 2 --gpu 1 --n_frames 100
```

Similarly, you can train Tensor4D for multi-view cases according to the following scripts:

```
# Train Tensor4D without flow
python exp_runner.py --case thumbsup_v4 --mode train --conf confs/t4d_origin.conf --gpu 0
```

After about 50k iterations of training, you can achieve a reasonably good result. If you want higher quality results, you may need to train for a longer period of time with more iterations, such as 200k iterations.

### Run Tensor4D with image guidance

Tensor4D can be further accelerated with image guidance. Here we provide a naive implementation which directly uses the 2D CNN to extract image features as additional conditions:

```
# Train Tensor4D with image guidance on thumbsup_v4
python exp_runner.py --case thumbsup30 --mode train --conf confs/t4d_thumbsup_img.conf --gpu 0

# Train Tensor4D with image guidance on dance_v4
python exp_runner.py --case dance_v4 --mode train --conf confs/t4d_dance_img.conf --gpu 0

# Train Tensor4D with image guidance on boxing_v12
python exp_runner.py --case boxing_v12 --mode train --conf confs/t4d_boxing_img.conf --gpu 0
```

### Config documentation

We provide the [config documentation](CONFIG.md) to explain the parameters in Tensor4D. It is recommended to check out the documentation before training your own Tensor4D model.

## Tensor4D Dataset

We provide Tensor4D dataset in this [link](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/shaorz20_mails_tsinghua_edu_cn/EsNxn0pJ19lFrRMKAS1YDx0Bv_V9LAdub9jnYvT40QZEDA?e=ChbsFX). Our dataset contains 5 multi-view sequences which is captured by 6 RGB cameras. All cameras are directed towards the front of the human. We will provide the scripts to process these raw data and convert them into our training samples.

We now provide the scripts to process raw data and convert them into our training samples in `scripts`. Thanks, Sun(286668458@qq.com), for writing and providing the data processing code.

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{shao2023tensor4d,
title = {Tensor4D: Efficient Neural 4D Decomposition for High-fidelity Dynamic Reconstruction and Rendering},
author = {Shao, Ruizhi and Zheng, Zerong and Tu, Hanzhang and Liu, Boning and Zhang, Hongwen and Liu, Yebin},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
year = {2023}
}
```

## Acknowledgments

Our project is benefit from these great resources:

- [NeuS:Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction.](https://github.com/Totoro97/NeuS)
- [TensoRF: Tensorial Radiance Fields](https://github.com/apchenstu/TensoRF)
- [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://github.com/bebeal/mipnerf-pytorch)

Thanks for their sharing code.