import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import lpips
import cv2 as cv
import os
import argparse
from tqdm import trange
import math

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def compute_psnr(image, gt):
    mse = img2mse(image, gt)
    psnr = mse2psnr(mse)
    return psnr

def gaussian(w_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
    return gauss/gauss.sum()

def create_window(w_size, channel=1):
    _1D_window = gaussian(w_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
    return window


def compute_ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)
    channel = img1.size(1)
    window = create_window(window_size, channel)
    window = window.to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(dim=1).mean()


def compute_lpips(img1, img2):
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)
    loss_fn = lpips.LPIPS(net='vgg').cuda()

    return loss_fn(img1, img2)
