# code required for resizing the images read
import os
import os.path as osp 
from glob import glob
from tqdm import tqdm
import argparse

import cv2

def resize_frame(image, resize_dim):
    center = image.shape[0]/2, image.shape[1]/2

    lower_dim = min(image.shape[0], image.shape[1])

    x = center[1] - lower_dim / 2
    y = center[0] - lower_dim / 2

    cropped_image = image[int(y):int(y+lower_dim), int(x):int(x+lower_dim)]

    # resize to the dimension - 256 x 256
    resized = cv2.resize(cropped_image, (resize_dim, resize_dim), interpolation=cv2.INTER_LINEAR)

    return resized    

def resize_frames(frames, resize_dim):
    resized = list()
    for frame in frames:
        resized.append(resize_frame(frame, resize_dim))

    return resized

def read_frame(frame_path):
    return cv2.imread(frame_path)

def read_frames(frame_paths):
    frames = list()
    for frame_path in frame_paths:
        frames.append(read_frame(frame_path))

    return frames

def save_frame(frame_path, frame):
    cv2.imwrite(frame_path, frame)

def save_frames(save_dir, frames):
    for index, frame in enumerate(frames):
        frame_path = osp.join(save_dir, str(index).zfill(3) + '.png')
        save_frame(frame_path, frame)


def process_dataset(source_dir, target_dir, resize_dim=128):
    directories = glob(source_dir + '/*')
    for dirname in tqdm(directories):
        # print(dirname)
        # read the frames from the source directory
        frame_paths = sorted(glob(dirname + '/*.jpg'))

        frames = read_frames(frame_paths)
        resized_frames = resize_frames(frames, resize_dim)

        # save the resized_frames to the disk
        current_target_dir = osp.join(target_dir, osp.basename(dirname))
        os.makedirs(current_target_dir, exist_ok=True)

        save_frames(current_target_dir, resized_frames)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str)
    parser.add_argument('--output_dir', default=None, type=str)

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    resize_dim = 128

    if output_dir is None:
        output_dir = osp.join(input_dir.rsplit('/', 1)[0], osp.basename(input_dir) + '_' + str(resize_dim))

    print(f'Output dir is : {output_dir}')

    process_dataset(input_dir, output_dir, resize_dim)

if __name__ == '__main__':
    main()