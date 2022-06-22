"""Generates a dataset of images using pretrained network pickle."""
import math
import sys; sys.path.extend(['.', 'src'])
import os
import os.path as osp
import random

import click
import dnnlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
# import skvideo

# skvideo.setFFmpegPath('/home2/aditya1/miniconda3/lib/python3.9/site-packages/ffmpeg')

# import skvideo.io

import skimage
import skimage.io

from PIL import Image

from glob import glob
from tqdm import tqdm

import legacy
from training.networks import Generator
from scripts import save_image_grid
from einops import rearrange

torch.set_grad_enabled(False)

# spherical linear interpolation (slerp)
def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0-val) * low + val * high
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
    
# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, num_interpolate_points):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=num_interpolate_points)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = slerp(ratio, p1, p2)
        vectors.append(v)
    return vectors

# method to generate the latent vectors in the dimension required by digan generator 
def generate_latents(num_videos, G, device):
    # z_values = torch.randn(num_videos, G.z_dim, device=device)
    # probably this is generating inference on the gpu
    print(f'Generator latent dimension is : {G.z_dim}')
    p1 = torch.randn(G.z_dim, device='cpu')
    p2 = torch.randn(G.z_dim, device='cpu')

    # generate num_videos points between p1 and p2 
    interpolated_latents = interpolate_points(p1, p2, num_videos)
    all_z = torch.vstack(interpolated_latents).to(device)

    z_tensors = all_z.split(1)

    return z_tensors

# method to generate the latent vectors in the dimension required by digan generator 
def generate_latents_zeros_ones(num_videos, G, device):
    # z_values = torch.randn(num_videos, G.z_dim, device=device)
    # probably this is generating inference on the gpu
    print(f'Generator latent dimension is : {G.z_dim}')
    # p1 = torch.zeros(G.z_dim, device='cpu')
    # p2 = torch.ones(G.z_dim, device='cpu')
    p1 = torch.randn(G.z_dim, device='cpu')
    p_ = torch.randn(G.z_dim, device='cpu')
    p2 = p_.clone()
    

    # generate num_videos points between p1 and p2 
    interpolated_latents = interpolate_points(p1, p2, num_videos)
    all_z = torch.vstack(interpolated_latents).to(device)

    z_tensors = all_z.split(1)

    return z_tensors

def save_as_frames(dir_path, frames):
    os.makedirs(dir_path, exist_ok=True)
    for index, frame in enumerate(frames):
        filepath = osp.join(dir_path, str(index).zfill(3) + '.png')

        skimage.io.imsave(filepath, frame)


@torch.no_grad()
def load_target_images(target_images_dir):
    target_image_paths = sorted(glob(target_images_dir + '/*.jpg'))
    images = [Image.open(f) for f in tqdm(target_image_paths, desc='Loading images')][:16]

    # images = [x[:, 200:-400, 450:-200] for x in images]
    images = [TVF.to_tensor(x) for x in images]
    images = [TVF.resize(x, size=(128, 128)) for x in images]

    return images

@torch.no_grad()
# method to read the images 
def read_target_image_features(target_images_dir, vgg16, device):
    target_images_path = sorted(glob(target_images_dir + '/*.jpg'))

    images = [Image.open(f) for f in tqdm(target_images_path, desc='Loading target images')][:16]

    images = [TVF.to_tensor(x) for x in images]
    images = [TVF.resize(x, size=(256, 256)) for x in images]

    # convert to torch tensor and get target features 
    target_features = []
    for img in images:
        
        img = img.to(device).to(torch.float32).unsqueeze(0) * 255.0
        # if img.shape[2] > 256:
        #     img = F.interpolate(img, size=(256, 256), mode='area')
        img = F.interpolate(img, size=(128, 128), mode='area')
        target_features.append(vgg16(img, resize_images=False, return_lpips=True).squeeze(0))
    target_features = torch.stack(target_features) # [num_images, lpips_dim]

    return target_features

@click.command()
@click.pass_context
@click.option('--network_pkl', help='Network pickle filename', required=True)
@click.option('--timesteps', type=int, help='Timesteps', default=16, show_default=True)
@click.option('--num_videos', type=int, help='Number of images to generate', default=100, show_default=True)
@click.option('--seed', type=int, help='Random seed', default=42, metavar='DIR')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
# @click.option('--target_images_dir', help='Directory of videos', type=str, required=True)
def generate_videos(
    ctx: click.Context,
    network_pkl: str,
    timesteps: int,
    num_videos: int,
    seed: int,
    outdir: str
):

    outdir, target_images_dir = outdir.split(":")

    os.makedirs(outdir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    target_images = load_target_images(target_images_dir)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
        G.forward = Generator.forward.__get__(G, Generator)
        
        G = G.requires_grad_(False).to(device)

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    target_features = []
    for img in target_images:   
        img = img.to(device).to(torch.float32).unsqueeze(0) * 255.0
        if img.shape[2] > 256:
            img = F.interpolate(img, size=(256, 256), mode='area')
        target_features.append(vgg16(img, resize_images=False, return_lpips=True).squeeze(0))
    target_features = torch.stack(target_features) # [num_images, lpips_dim]

    grid_size = (int(math.sqrt(num_videos)), int(math.sqrt(num_videos)))

    # this is the grid of z values -- for a single video -- one z is required?
    z = torch.randn([1, G.z_dim], device=device, requires_grad=True)
    z_motion = torch.randn([1, 512], device=device, requires_grad=True)

    # z = torch.tensor(z).detach().to(device).requires_grad_(True)
    # z_motion = torch.tensor(z_motion).detach().to(device).requires_grad_(True)

    z = z.clone().detach().requires_grad_(True)
    z_motion = z_motion.clone().detach().requires_grad_(True)

    # z = z.detach().requires_grad_(True)
    # z_motion = z_motion.detach().requires_grad_(True)

    # grid_z = torch.randn([int(grid_size[0] * grid_size[1]), G.z_dim], device=device, requires_grad=True).split(1)
    # grid_motion = torch.randn([int(grid_size[0] * grid_size[1]), 512], device=device, requires_grad=True).split(1)
    
    initial_learning_rate = 25e-4

    # optimizer = torch.optim.Adam([grid_z] + [grid_motion], betas=(0.9, 0.999), lr=initial_learning_rate)
    optimizer = torch.optim.Adam([z] + [z_motion], betas=(0.9, 0.999), lr=initial_learning_rate)

    print(f'z_motion shape : {z_motion.shape}')

    for step in tqdm(range(1000)):
        synth_images = G(z, None, z_motion=z_motion, timesteps=16, noise_mode='const')[0]
        print(f'Synth images min and max are : {torch.min(synth_images), torch.max(synth_images)}')

        # check for params for G
        for param in G.parameters():
            print(param.requires_grad)

        # read the synth features and generate the features
        synth_images = (synth_images * 0.5 + 0.5) * 255.0
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        print(f'synth features shape : {synth_features.shape}, target features shape : {target_features.shape}')
        dist = (target_features - synth_features).square().sum()

        # # check params of synth_features
        # print(f'Synth feature grad : {synth_features.requires_grad}')
        # assert not True

        motion_reg_type = None # was set by default in the stylegan-v repository

        # Noise regularization.
        if motion_reg_type is None:
            reg_loss = 0.0
        else:
            raise NotImplementedError(f"Uknown motion_reg_type: {motion_reg_type}")

        regularize_noise_weight = 1e-4

        loss = dist
        
        # check gradients of z and z_motion 
        print(f'z gradients : {z.requires_grad}, z_motion gradients : {z_motion.requires_grad}')
        print(z_motion)

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print(f'Images shape : {images.shape}')

    for index, video in enumerate(images):
        permuted = video.transpose(1, 2, 3, 0)

        # save as frames
        dir_path = osp.join(outdir, str(index).zfill(3))
        print(f'Saving images to dir : {dir_path}')
        save_as_frames(dir_path, permuted)

    save_image_grid(images, os.path.join(outdir, f'generate_videos.gif'), drange=[-1, 1], grid_size=grid_size)


if __name__ == "__main__":
    generate_videos()