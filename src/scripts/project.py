"""Generates a dataset of images using pretrained network pickle."""
import math
import sys

from nbformat import current_nbformat; sys.path.extend(['.', 'src'])
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

import skimage
import skimage.io

from PIL import Image

from glob import glob
from tqdm import tqdm

import legacy
from training.networks import Generator
from scripts import save_image_grid
from einops import rearrange

# torch.set_grad_enabled(False)

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
    print(f'Generator latent dimension is : {G.z_dim}')
    p1 = torch.randn(G.z_dim, device='cpu')
    p_ = torch.randn(G.z_dim, device='cpu')
    p2 = p_.clone()
    

    # generate num_videos points between p1 and p2 
    interpolated_latents = interpolate_points(p1, p2, num_videos)
    all_z = torch.vstack(interpolated_latents).to(device)

    z_tensors = all_z.split(1)

    return z_tensors

def save_as_frames(dir_path, frames, type):
    os.makedirs(dir_path, exist_ok=True)
    for index, frame in enumerate(frames):
        filepath = osp.join(dir_path, str(index).zfill(3) + '_{}.png'.format(type))

        skimage.io.imsave(filepath, frame)


@torch.no_grad()
def load_target_images(target_images_dir):
    target_image_paths = sorted(glob(target_images_dir + '/*.png'))
    images = [Image.open(f) for f in tqdm(target_image_paths, desc='Loading images')]

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
# @click.option('--num_videos', type=int, help='Number of images to generate', default=100, show_default=True)
@click.option('--seed', type=int, help='Random seed', default=42, metavar='DIR')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
# @click.option('--target_images_dir', help='Directory of videos', type=str, required=False, metavar='DIR')
@click.option('--num_steps', type=int, help='Optimization steps', default=1000, show_default=True)
@click.option('--save_steps', type=int, help='Steps after which to save', default=50, show_default=True)
@click.option('--loss_type', help='Type of loss function to optimize', default='perceptual', required=True)
@click.option('--dir_videos', help='Directory of all videos', type=str, required=False, metavar='DIR')
@click.option('--task', help='Type of inversion task', default=None, required=False)
@click.option('--save_target', help='Save target images', default=False, type=bool, metavar='BOOL')
@click.option('--extrapolate_frames', type=int, help='Frames required for extrapolation', default=4, show_default=True)

def generate_videos(
    ctx: click.Context,
    network_pkl: str,
    timesteps: int,
    # num_videos: int,
    seed: int,
    outdir: str,
    # target_images_dir,
    num_steps: int,
    save_steps: int,
    loss_type: str,
    dir_videos: str,
    task: str,
    save_target: bool,
    extrapolate_frames: int
):
    w_avg_samples = 10000
    num_videos = 1
    initial_learning_rate = 25e-4

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda')

    ''' Common code
    '''
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
        G.forward = Generator.forward.__get__(G, Generator)
        
        G = G.eval().requires_grad_(False).to(device)

    if loss_type == 'perceptual':
        print(f'Require perceptual loss')
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

    ''' Common code
    '''

    dirs = glob(dir_videos + '/*')
    print(f'Total number of dirs to process : {len(dirs)}')

    if task == 'sparse':
        # create the hidden pixels for the task of sparse inpainting 
        max_value = 128*128*timesteps
        indices = torch.randperm(max_value) 
        assert len(indices) == max_value

        sparse_sampling_factor = 0.75
        hidden_indices = indices[:int(sparse_sampling_factor*max_value)]
        loss_indices = indices[int(sparse_sampling_factor*max_value):]

    for dirname in tqdm(dirs):
        # create the outdir from the input dir 
        current_outdir = osp.join(outdir, osp.basename(dirname))
        os.makedirs(current_outdir, exist_ok=True)

        target_images = load_target_images(dirname)

        z = torch.randn([1, G.z_dim], device=device, requires_grad=True) # 1 x 512
        z_motion = torch.randn([1, 512], device=device, requires_grad=True) # 1 x 512

        optimizer = torch.optim.Adam([z] + [z_motion], betas=(0.9, 0.999), lr=initial_learning_rate)

        target_images = torch.vstack(target_images).to(device)

        # normalize the range of target images (current -> [0, 1], required -> [-1, 1])
        target_images = (target_images - 0.5) / 0.5 # [-1, 1]

        for step in tqdm(range(num_steps)):
            synth_images = G(z, None, z_motion=z_motion, timesteps=timesteps, noise_mode='const')[0]
            
            if loss_type == 'perceptual':
                # Features for synth images.
                synth_images = (synth_images * 0.5 + 0.5) * 255.0
                if synth_images.shape[2] > 256:
                    synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
                synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)

            synth_images = synth_images.view(-1, synth_images.shape[-2], synth_images.shape[-1])
            assert synth_images.shape == target_images.shape

            if loss_type == 'perceptual':
                loss = (target_features - synth_features).square().mean()
            else:
                print(f'Using L1 distance loss')
                print(f'Task : {task}')
                # for inpainting (upper-half context given -- the images need to be lower-half masked)
                if task == 'inpainting':
                    # inpainting is an inversion task 
                    # input frames shape -> num_frames x channels x image_dim x image_dim
                    print(f'Shapes before inpainting : target - {target_images.shape}, {synth_images.shape}')

                    # print(f'Unique target values in task : {torch.unique(target_images)}')
                    # print(f'Unique synth values in task : {torch.unique(synth_images)}')

                    # mask the lower half of the target image - set to white (255)
                    target_modified = target_images.clone()
                    target_modified[:,64:] = 1.0

                    # compare the top half of target with top half of synthesized 
                    # L1 distance loss
                    loss = torch.abs(target_modified[:, :64] - synth_images[:, :64]).mean()

                elif task == 'extrapolation':
                    # in extrapolation, only the first four frames are used as the input
                    print(f'Using {extrapolate_frames} frames to extrapolate')
                    target_modified = target_images[:extrapolate_frames*3]

                    # compute the L1 loss
                    loss = torch.abs(target_modified - synth_images[:extrapolate_frames*3]).mean()

                elif task == 'interpolation':
                    # in interpolation, the first and the last frame would be used as the context
                    print(f'Using the first and the last frame for interpolation')

                    target_modified = torch.vstack([target_images[:3], target_images[-3:]])

                    # compute the L1 loss
                    loss = torch.abs(target_modified - torch.vstack([synth_images[:3], synth_images[-3:]])).mean()

                elif task == 'sparse':
                    # in the optimization objective, the supervision is only from (1-sparse_sampling_factor) of the image
                    # target_images.shape -> (25*3) x 128 x 128 -> n x 3
                    target_modified = target_images.view(-1, 3, 128, 128).permute(0, 2, 3, 1).reshape(-1, 3)
                    synth_modified = synth_images.view(-1, 3, 128, 128).permute(0, 2, 3, 1).reshape(-1, 3)

                    # compute the loss on the pixels corresponding to loss_indices 
                    loss = torch.abs(target_modified[loss_indices] - synth_modified[loss_indices]).mean()
                    
                    # set the pixels corresponding to hidden_indices to white (1)
                    target_modified[hidden_indices] = 1

                    # initial -> (25*128*128) x 3 -> (25*3) x 128 x 128
                    target_modified = target_modified.view(-1, 128, 128, 3).permute(0, 3, 1, 2).reshape(-1, 128, 128)

                else:
                    # Computing the normal L1 loss
                    loss = torch.abs(target_images - synth_images).mean()
            
            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            print(loss)

            def save_step(synth_images, target_images, target_modified, steps, current_outdir, task):
                frame_dims = synth_images.shape[-1]

                print(f'Synth images shape : {synth_images.shape}')

                # the synth images need to be normalized using the respective code
                synth_images.clamp(-1, 1)
                synth_images = synth_images.view(-1, 3, frame_dims, frame_dims) # time_steps x 3 x 128 x 128
                synth_images = synth_images.permute(0, 2, 3, 1).detach().cpu() # time_steps x 128 x 128 x 3

                current_output_dir = osp.join(current_outdir, str(steps).zfill(4))
                print(f'Saving images to dir : {current_output_dir}')

                save_as_frames(current_output_dir, synth_images, 'synth')

                # save the target images as well
                if save_target:
                    # print(f'Unique target_images values : {torch.unique(target_images)}')
                    target_images.clamp(-1, 1)
                    target_images = target_images.view(-1, 3, frame_dims, frame_dims)
                    target_images = target_images.permute(0, 2, 3, 1).detach().cpu()

                    # save the original target image and the masked image 
                    save_as_frames(current_output_dir, target_images, 'target')

                    if task is not None:
                        # print(f'Unique target_lower_masked values : {torch.unique(target_lower_masked)}')
                        target_modified.clamp(-1, 1)
                        target_modified = target_modified.view(-1, 3, frame_dims, frame_dims)
                        target_modified = target_modified.permute(0, 2, 3, 1).detach().cpu()

                        save_as_frames(current_output_dir, target_modified, 'target_modified')

            if (step+1)%save_steps == 0:
                # save synth_images
                if task is not None:
                    save_step(synth_images, target_images, target_modified, (step+1), current_outdir, task)
                else:
                    save_step(synth_images, target_images, None, (step+1), current_outdir, task)

        # save image grid if saving as grid is required
        # save_image_grid(synth_images.detach().cpu().numpy(), os.path.join(outdir, f'generate_videos.gif'), drange=[-1, 1], grid_size=grid_size)


if __name__ == "__main__":
    generate_videos()