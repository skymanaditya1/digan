# this is the model which takes in a series of snapshot files as input and generates the inference 
import os
from glob import glob
import os.path as osp

styleganv_dir = '/ssd_scratch/cvit/aditya1/stylegan-v'
digan_dir = '/ssd_scratch/cvit/aditya1/digan'

def generate_videos(gpu_id, checkpoint, results_dir, num_videos):
    command_format = 'CUDA_VISIBLE_DEVICES={} python src/scripts/generate_videos.py \
            --network_pkl {} \
            --outdir {} \
            --num_videos {}'

    command = command_format.format(gpu_id, checkpoint, results_dir, num_videos)

    print(f'Running command : {command}')

    output = os.popen(command).read()

    print(output)


def read_checkpoints(checkpoint_dir):
    # reads all checkpoints from the dir 
    checkpoints = glob(checkpoint_dir + '/*.pkl')

    return checkpoints

def main():

    model_name = 'digan'
    dataset = 'how2sign_faces'

    input_dir = '/ssd_scratch/cvit/aditya1/important_checkpoints/{}/{}'.format(model_name, dataset)
    gpu_id = 1
    num_videos = 2

    os.chdir(digan_dir)

    # results dir would depend on the checkpoint
    checkpoints = read_checkpoints(input_dir)
    for checkpoint in checkpoints:
        results_dir = osp.join(input_dir, osp.basename(checkpoint).split('.')[0])
        os.makedirs(results_dir, exist_ok=True)
        generate_videos(gpu_id, checkpoint, results_dir, num_videos)

if __name__ == '__main__':
    main()