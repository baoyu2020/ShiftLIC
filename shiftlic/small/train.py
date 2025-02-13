import argparse
import math
import random
import shutil
import sys
import os
import numpy as np
import time
import warnings
from sys import flags

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d, conv3x3, subpel_conv3x3
from compressai.models.utils import conv, deconv, update_registered_buffers

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,)

from pytorch_msssim import ms_ssim
from runx.logx import logx
from tensorboardX import SummaryWriter
from basic_layers import * 
from utils import Demultiplexer, Multiplexer, MultistageMaskedConv2d, quantize_ste, Round_with_grad
from ptflops import get_model_complexity_info

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(
        min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class ScaleHyperpriorsAutoEncoder(CompressionModel):
    """autoencoder with a ScaleHyperpriors from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018."""

    def __init__(self, N, M, **kwargs):
        super().__init__(N, **kwargs)
        ''' all shift with  CheapCS attention '''
        self.encoder = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(12, 128, kernel_size=1, stride=1, padding=0),
            ResidualBlockShift(128, 128),
            ResidualBlockShift(128, 128),
            ResidualBlockShift(128, 128),
            nn.PixelUnshuffle(2),
            nn.Conv2d(128*4, 192, kernel_size=1, stride=1, padding=0),
            ResidualBlockShift(192, 192),
            ResidualBlockShift(192, 192),
            ResidualBlockShift(192, 192),
            nn.PixelUnshuffle(2),
            nn.Conv2d(192*4, 256, kernel_size=1, stride=1, padding=0),
            ResidualBlockShift(256, 256),
            ResidualBlockShift(256, 256),
            ResidualBlockShift(256, 256),
            nn.PixelUnshuffle(2),
            nn.Conv2d(256*4, M, kernel_size=1, stride=1, padding=0),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(M, 256*4, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
            ResidualBlockShift(256, 256),
            ResidualBlockShift(256, 256),
            ResidualBlockShift(256, 256),
            nn.Conv2d(256, 192*4, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
            ResidualBlockShift(192, 192),
            ResidualBlockShift(192, 192),
            ResidualBlockShift(192, 192),
            nn.Conv2d(192, 128*4, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
            ResidualBlockShift(128, 128),
            ResidualBlockShift(128, 128),
            ResidualBlockShift(128, 128),
            nn.Conv2d(128, 12, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
        )
        self.hyperencoder = nn.Sequential(                
            ResidualBlockShift(M, N),
            nn.LeakyReLU(inplace=True),
            ResidualBlockShift(N, N),
            nn.LeakyReLU(inplace=True),                
            nn.PixelUnshuffle(2),
            nn.Conv2d(N*4, N, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            ResidualBlockShift(N, N),
            nn.LeakyReLU(inplace=True),
            nn.PixelUnshuffle(2),
            nn.Conv2d(N*4, N, kernel_size=1, stride=1, padding=0),
        )
        self.hyperdecoder = nn.Sequential(
            ResidualBlockShift(N, N),
            nn.LeakyReLU(inplace=True),                
            nn.Conv2d(N, N*4, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
            nn.LeakyReLU(inplace=True),
            ResidualBlockShift(N, N),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N*4, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
            nn.LeakyReLU(inplace=True),
            ResidualBlockShift(N, M),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
    
    def forward(self, x):
        y =  self.encoder(x)
        z = self.hyperencoder(torch.abs(y))
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset
        scales = self.hyperdecoder(z_hat)
        _, y_likelihoods = self.gaussian_conditional(y, scales)
        y_hat = quantize_ste(y)
        x_hat = self.decoder(y_hat)
        x_hat = x_hat.clamp(0., 1.)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods, "z": z_likelihoods},
            "y": y,
            "y_hat": y_hat,
            "scale": scales,
            "z": z
        }

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["encoder.0.weight"].size(0)
        M = state_dict["encoder.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        super().update(force=force)

    # for inference
    def compress(self, x):
        y = self.encoder(x)
        z = self.hyperencoder(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales = self.hyperdecoder(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.hyperdecoder(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        x_hat = self.decoder(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
    
#################################################################################################################
#################################################################################################################

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, metric="mse"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ms_ssim = ms_ssim
        self.lmbda = lmbda
        self.metric = metric

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["ms_ssim_loss"] = self.ms_ssim(output["x_hat"], target, data_range=1)
        if self.metric == "ms_ssim":
            distortion = 1 - out["ms_ssim_loss"]
        else:
            distortion = 255**2 * out["mse_loss"]
        out["loss"] = self.lmbda * distortion + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


#################################################################################################################
#################################################################################################################

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, writer, args
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)

        loss = out_criterion["loss"]
        loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10000 == 0:
            psnr_loss = 10 * np.log10(1.0 / out_criterion["mse_loss"].item())
            ms_ssim_loss = -10 * np.log10(1.0 - out_criterion["ms_ssim_loss"].item())
            logx.msg(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tRD Loss: {loss.item():.6f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.4f} |'
                f'\tPSNR: {psnr_loss:.4f} |'
                f'\tMSSSIM: {ms_ssim_loss:.4f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
        niter = epoch * len(train_dataloader) + i
        # writer.add_scalar('Train/RD_Loss', loss.item(), niter)
        logx.add_scalar('Train/RD_Loss', loss.item(), niter)


def test_epoch(epoch, test_dataloader, model, criterion, writer, args):
    model.eval()
    device = next(model.parameters()).device

    rd_loss = AverageMeter()
    bpp_loss = AverageMeter()
    psnr_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)  
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            aux_loss.update(model.aux_loss())
            rd_loss.update(out_criterion["loss"])
            bpp_loss.update(out_criterion["bpp_loss"])
            psnr_loss.update(10 * np.log10(1.0 / out_criterion["mse_loss"].item()))
            ms_ssim_loss.update(-10 * np.log10(1.0 - out_criterion["ms_ssim_loss"].item()))
    
    logx.msg(
        f"=========test kodak===================\n"
        f"Test epoch {epoch}: Average losses:"
        f"\tRD loss: {rd_loss.avg:.6f} |"
        f"\tBpp: {bpp_loss.avg:.4f} |"
        f'\tPSNR: {psnr_loss.avg:.4f} |'
        f'\tMSSSIM: {ms_ssim_loss.avg:.4f} |'
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    writer.add_scalar('Test/RD_Loss', rd_loss.avg, epoch)
    writer.add_scalar('Test/bpp', bpp_loss.avg, epoch)
    writer.add_scalar('Test/PSNR', psnr_loss.avg, epoch)
    writer.add_scalar('Test/MS-SSIM', ms_ssim_loss.avg, epoch)
    return rd_loss.avg

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-8]+"_best_loss"+filename[-8:])

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed every 40 epochs"""
    if epoch > 80:
        lr = init_lr * 0.1
    elif epoch > 40:
        lr = init_lr * 0.5
    else:
        lr = init_lr

    if lr < 1e-6:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-d", "--dataset", type=str, default='/private/data/CLIC2020', help="Training dataset")
    parser.add_argument("-e", "--epochs", default=100, type=int, help="Number of epochs (default: %(default)s)",)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)",)
    parser.add_argument("-n", "--num-workers", type=int, default=8, help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--lambda", dest="lmbda",type=float,default=0.100,help="Bit-rate distortion parameter (default: %(default)s)",)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size",type=int,default=1,help="Test batch size (default: %(default)s)",)
    parser.add_argument("--aux-learning-rate",default=1e-3,help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument('--gpu_id',type=int, default=0, help='The number of GPU')
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument('-m','--metric',type=str,default='mse',help='Optimized metric, choose from mse or msssim')
    parser.add_argument('--out_dir',type=str,default='./Log/stage_perf/',help='path of log and saved models')
    parser.add_argument("--seed", type=float, default=100, help="Set random seed for reproducibility")
    parser.add_argument("--clip_max_norm", default=1.0, type=float, help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

#################################################################################################################
#################################################################################################################

def main(argv):
    args = parse_args(argv)
    print(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    log_dir = os.path.join(args.out_dir, 'runlog/')
    logx.initialize(logdir=log_dir, coolname=False, tensorboard=False, hparams=vars(args), eager_flush=True)
    writer = SummaryWriter(log_dir)


    train_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="kodak", transform=test_transforms)

    # device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    device = torch.device("cuda", args.gpu_id)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = ScaleHyperpriorsAutoEncoder(192,320)

    macs, params = get_model_complexity_info(net, (3, 256, 256), print_per_layer_stat=True, as_strings=True, verbose=True)  
    logx.msg(f"macs:{macs}\t")
    logx.msg(f"params:{params}\t")
    logx.msg(f"networks:{net}\t")

    net = net.to(device)

    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metric=args.metric)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        # print("Loading", args.checkpoint)
        logx.msg(f"Loading:{args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        cur_lr = adjust_learning_rate(optimizer, epoch, args.learning_rate)
        # print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        logx.msg(f"Learning rate: {cur_lr}")
        st_time = time.time()
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            writer,
            args,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, writer, args,)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        logx.msg(f"epoch time: {time.time() - st_time}")

        save_dir = os.path.join(args.out_dir, 'ckpt/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                },
                is_best,
                os.path.join(save_dir, args.metric+str(args.lmbda)+'checkpoint.pth.tar')
            )


if __name__ == "__main__":
    main(sys.argv[1:])