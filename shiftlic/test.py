import argparse
import math
import os
import time
import sys
import glob
import json

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

from train import ScaleHyperpriorsAutoEncoder

def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(
    org: torch.Tensor, rec: torch.Tensor, max_val: int = 255
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["mse"] = nn.MSELoss()(org, rec).item()
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    # metrics["mse"] = nn.MSELoss()(org, rec).item()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics

@torch.no_grad()
def inference(model, x):
    # x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    if args.real_bit:
        model.update()
        start = time.time()
        out_enc = model.compress(x_padded)
        enc_time = time.time() - start

        start = time.time()
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        dec_time = time.time() - start
        metrics = compute_metrics(x_padded, out_dec["x_hat"], 255)
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    else:
        start = time.time()
        out_dec = model.forward(x_padded)
        dec_time = time.time() - start
        dec_time = dec_time / 2
        enc_time = dec_time
        metrics = compute_metrics(x_padded, out_dec["x_hat"], 255)
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        bpp = sum( (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_dec["likelihoods"].values()
        )
        bpp = bpp.item()

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    # input images are 8bit RGB for now


    return {
        "img": out_dec["x_hat"],
        "mse": metrics["mse"],
        "psnr": metrics["psnr"],
        "ms-ssim": metrics["ms-ssim"],
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

@torch.no_grad()
def inference_rd(im_dirs, args):
    device = args.device
    N, M = 192, 320
    net = ScaleHyperpriorsAutoEncoder(N, M)
    model_dict = torch.load(args.checkpoint, map_location='cuda', weights_only=True)
    print(model_dict["epoch"])
    net.load_state_dict(model_dict["state_dict"])
    net = net.eval().to(device)
    loss_list = []
    psnr_list = []
    msssim_list = []
    bpp_list = []
    enc_time_list = []
    dec_time_list = []
    for im_dir in im_dirs:
        im_name = os.path.basename(im_dir)
        pic_index = im_name[:-4]
        rec_dir = os.path.join(args.output, 'img_dec' + str(pic_index) + '.png')
        img = Image.open(im_dir)
        ori_img = np.array(img)
        test_transforms = transforms.Compose([transforms.ToTensor()])
        img = test_transforms(img)
        _, H, W = img.shape
        img = img.unsqueeze(0).to(device)
        print('====> Encoding Image:', im_dir)
        with torch.no_grad():            
            output = inference(net, img)

        if args.metric == "ms_ssim":
            distortion = 1 - output["ms-ssim"]
        else:
            distortion = output["mse"]
        rdloss = args.lmbda * distortion + output["bpp"]
        loss_list.append(rdloss)

        psnr_list.append(output["psnr"])
        msssim_list.append(output["ms-ssim"])
        bpp_list.append(output["bpp"])
        enc_time_list.append(output["encoding_time"])
        dec_time_list.append(output["decoding_time"])

        if args.save:
            out_img = output["img"].clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_img = np.round(out_img * 255.0).astype('uint8')
            out_img = out_img.astype('uint8')
            dec_img = Image.fromarray(out_img)
            dec_img.save(rec_dir)
            del out_img

        # 保存每张图像的测试结果
        image_result = {
            "image_name": im_name,
            "mse": output["mse"],
            "psnr": output["psnr"],
            "ms-ssim": output["ms-ssim"],
            "bpp": output["bpp"],
            "encoding_time": output["encoding_time"],
            "decoding_time": output["decoding_time"]
        }
        image_result_path = os.path.join(args.output, f"img_result_{pic_index}.json")
        with open(image_result_path, 'w') as f:
            json.dump(image_result, f, indent=4)
    
    mean_loss = np.mean(loss_list)
    mean_bpp = np.mean(bpp_list)
    mean_psnr = np.mean(psnr_list)
    mean_msssim = np.mean(msssim_list)
    mean_enc_time = np.mean(enc_time_list)
    mean_dec_time = np.mean(dec_time_list)
    mean_msssim_dB = -10 * np.log10(1.0 - mean_msssim)

    print("mean of loss:", mean_loss)
    print("mean of bpp:", mean_bpp)
    print("mean of psnr:", mean_psnr)
    print("mean of msssim:", mean_msssim)
    print("mean of msssim_dB:", mean_msssim_dB) 
    print("mean of enc_time:", mean_enc_time)
    print("mean of dec_time:", mean_dec_time)

    log_name = os.path.join(args.output, str(args.lmbda)+'RD.log')
    with open(log_name, 'a') as f:
        f.write('========Testing========='+ '\n'+"mean of loss:" + str(mean_loss) + '\n' + "mean of bpp:" + str(mean_bpp) + '\n' + "mean of psnr:" + str(mean_psnr) +'\n' + "mean of msssim:" + str(mean_msssim)+ '\n' + "mean of mean_msssim_dB:"+ str(mean_msssim_dB) +'\n' + "mean of enc_time:" + str(mean_enc_time)+'\n' + "mean of dec_time:" + str(mean_dec_time) + '\n'+ "mean of latency:" + str(1000*(mean_enc_time+mean_dec_time)) + '\n')
    f.close()    
    
    # 保存整个数据集的测试结果
    dataset_result = {
        "mean_loss": mean_loss,
        "mean_bpp": mean_bpp,
        "mean_psnr": mean_psnr,
        "mean_msssim": mean_msssim,
        "mean_msssim_dB": mean_msssim_dB,
        "mean_enc_time": mean_enc_time,
        "mean_dec_time": mean_dec_time
    }
    dataset_result_path = os.path.join(args.output, "dataset_result.json")
    with open(dataset_result_path, 'w') as f:
        json.dump(dataset_result, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default='/data1/home/ynbao/data/jpeg-ai/', help="Input Image")
    parser.add_argument("-o", "--output", type=str, default="./Output/JPEG", help="Output Bin(encode)/Image(decode)")
    parser.add_argument("--checkpoint", type=str, default='/data1/home/ynbao/Testcodec/deep_models/ELIC/output/Q1/ckpt/mse0.0018checkpoint_best_loss.pth.tar', help="Path to a checkpoint")
    parser.add_argument("-m", "--metric", type=str, default='ms_ssim')
    parser.add_argument("--lambda", dest="lmbda",type=float,default=0.100,help="Bit-rate distortion parameter (default: %(default)s)",)
    # 6.51, 8.73, 16.64, 31.73, 60.5, 140.0
    parser.add_argument("--save", action='store_true', help="save the decoded image")
    parser.add_argument("--real_bit", action='store_true', default=False , help="use real bit or not")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use: 'cuda' or 'cpu'")
    args = parser.parse_args()
    print(args)
    # logx.initialize(logdir=args.output, coolname=False, tensorboard=False, hparams=vars(args), eager_flush=True)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    test_images = []
    if os.path.isdir(args.input):
        dirs = os.listdir(args.input)
        for dir in dirs:
            path = os.path.join(args.input, dir)
            if os.path.isdir(path):
                test_images += glob.glob(path + '/*.png')
            if os.path.isfile(path):
                test_images.append(path)
    else:
        test_images.append(args.input)

    inference_rd(test_images, args)