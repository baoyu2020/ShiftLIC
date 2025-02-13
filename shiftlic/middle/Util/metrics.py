import math

import numpy as np
import torch

from Util.msssim import MultiScaleSSIM as msssim_
import Util.torch_msssim as torch_msssim
# numpy
def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX**2 / mse)

# pytorch
def torch_rgb2yuv444(img):
    # input:    torch tensor [N, 3, H, W]
    # output:   torch tensor [N, 3, H, W]
    ycbcr = torch.zeros_like(img, dtype=torch.float32)

    r = img[:, 0, :, :]
    g = img[:, 1, :, :]
    b = img[:, 2, :, :]

    # BT.601
    convert_mat = np.array([[0.299, 0.587, 0.114],
                            [-0.1687, -0.3313, 0.5],
                            [0.5, -0.4187, -0.0813]], dtype=np.float32)

    ycbcr[:, 0, :, :] = r * convert_mat[0, 0] + \
        g * convert_mat[0, 1] + b * convert_mat[0, 2]
    ycbcr[:, 1, :, :] = r * convert_mat[1, 0] + g * \
        convert_mat[1, 1] + b * convert_mat[1, 2] + 128.
    ycbcr[:, 2, :, :] = r * convert_mat[2, 0] + g * \
        convert_mat[2, 1] + b * convert_mat[2, 2] + 128.

    return ycbcr

def mse_yuv444(yuv_0, yuv_1):
    psnr_weights = [6.0/8.0, 1.0/8.0, 1.0/8.0]

    mse = 1.
    for i in range(3):
        mse_w = torch.mean((yuv_1[:, i, :, :] - yuv_0[:, i, :, :])**2.0)
        # psnr = 10.0 * torch.log10(255.*255./mse) * psnr_weights[i] + psnr
        mse = mse * (mse_w ** psnr_weights[i])
        print(mse_w)
    return mse

def msssim_yuv444(yuv_0, yuv1):
    # TODO：Y MS-SSIM
    MS_SSIM_ = torch_msssim.MS_SSIM(max_val=1.)
    ms_ssim_yuv = MS_SSIM_.ms_ssim(yuv_0, yuv1)
    return ms_ssim_yuv

class YUV_MS_SSIM_Loss(torch.nn.Module):
    def __init__(self):
        super(YUV_MS_SSIM_Loss, self).__init__()

    def forward(self, im0, im1):
        yuv_0 = torch_rgb2yuv444(im0)
        yuv_1 = torch_rgb2yuv444(im1)
        return msssim_yuv444(yuv_0, yuv_1)

class YUV_MSELoss(torch.nn.Module):
    def __init__(self):
        super(YUV_MSELoss, self).__init__()

    def forward(self, im0, im1):
        yuv_0 = torch_rgb2yuv444(im0)
        yuv_1 = torch_rgb2yuv444(im1)
        return mse_yuv444(yuv_0, yuv_1)

def rgb2yuv444(img):
    img = np.array(img, dtype=np.float32)
    ycbcr = np.zeros_like(img, dtype=np.float32)

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    convert_mat = np.array([[0.299, 0.587, 0.114],
							[-0.1687, -0.3313, 0.5],
							[0.5, -0.4187, -0.0813]], dtype=np.float32)

    ycbcr[:, :, 0] = r * convert_mat[0, 0] + g * convert_mat[0, 1] + b * convert_mat[0, 2]
    ycbcr[:, :, 1] = r * convert_mat[1, 0] + g * convert_mat[1, 1] + b * convert_mat[1, 2] + 128.
    ycbcr[:, :, 2] = r * convert_mat[2, 0] + g * convert_mat[2, 1] + b * convert_mat[2, 2] + 128.

    return ycbcr

def psnr_yuv444(yuv_0, yuv_1):
    psnr_weights = [6.0/8.0, 1.0/8.0, 1.0/8.0]

    psnr = 0.
    for i in range(3):
        mse = np.mean(np.square(yuv_1[:, :, i] - yuv_0[:, :, i]))
        psnr = 10.0 * np.log10(255.*255./mse) * psnr_weights[i] + psnr

    return psnr

def evaluate(img0, img1):
    img0 = img0.astype('float32')
    img1 = img1.astype('float32')
    H, W, C = img0.shape
    rgb_psnr = psnr(img0, img1)
    # 图像维度： batch * H * W * C
    imgs0 = np.zeros((1, H, W, C))
    imgs1 = np.zeros((1, H, W, C))
    imgs0[0], imgs1[0] = img0, img1
    r_msssim = msssim_(imgs0[:, :, :, 0].reshape(1,H,W,1), imgs1[:, :, :, 0].reshape(1,H,W,1))
    g_msssim = msssim_(imgs0[:, :, :, 1].reshape(1,H,W,1), imgs1[:, :, :, 1].reshape(1,H,W,1))
    b_msssim = msssim_(imgs0[:, :, :, 2].reshape(1,H,W,1), imgs1[:, :, :, 2].reshape(1,H,W,1))
    rgb_msssim = (r_msssim + g_msssim + b_msssim)/3
    yuv0 = rgb2yuv444(img0)
    yuv1 = rgb2yuv444(img1)
    yuv_psnr = psnr_yuv444(yuv0, yuv1)
    H, W, C = yuv0.shape
    yuvs0 = np.zeros((1, H, W, C))
    yuvs1 = np.zeros((1, H, W, C))
    yuvs0[0], yuvs1[0] = yuv0, yuv1
    y_msssim = msssim_(yuvs0[:, :, :, 0].reshape(1,H,W,1), yuvs1[:, :, :, 0].reshape(1,H,W,1))
    return rgb_psnr, rgb_msssim, yuv_psnr, y_msssim