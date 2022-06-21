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
# import skvideo

# skvideo.setFFmpegPath('/home2/aditya1/miniconda3/lib/python3.9/site-packages/ffmpeg')

# import skvideo.io

import skimage
import skimage.io

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

@click.command()
@click.pass_context
@click.option('--network_pkl', help='Network pickle filename', required=True)
@click.option('--timesteps', type=int, help='Timesteps', default=16, show_default=True)
@click.option('--num_videos', type=int, help='Number of images to generate', default=100, show_default=True)
@click.option('--seed', type=int, help='Random seed', default=42, metavar='DIR')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
# @click.option('--interpolate', help='Whether interpolation is required', action='store_true')
def generate_videos(
    ctx: click.Context,
    network_pkl: str,
    timesteps: int,
    num_videos: int,
    seed: int,
    outdir: str
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore

        # set the generator image resolution explicitly 
        # G.img_resolution = 256
        # G.synthesis.img_resolution = 256
        print(f'Generator image resolution is : {G.img_resolution}')

        G.forward = Generator.forward.__get__(G, Generator)
        print("Done. ")
        assert not True

    os.makedirs(outdir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    grid_size = (int(math.sqrt(num_videos)), int(math.sqrt(num_videos)))

    # this is the grid of z values -- for a single video -- one z is required?
    interpolate = True
    if interpolate:
        print(f'Required interpolation')
        required = int(math.sqrt(num_videos)) ** 2

        grid_z = generate_latents(required, G, device)
    else:
        grid_z = torch.randn([int(grid_size[0] * grid_size[1]), G.z_dim], device=device).split(1)

    images = torch.cat([rearrange(
                        G(z, None, timesteps=16, noise_mode='const')[0].cpu(),
                        '(b t) c h w -> b c t h w', t=timesteps) for z in grid_z]).numpy()        

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
