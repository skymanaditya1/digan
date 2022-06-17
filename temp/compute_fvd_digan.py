# This code is used for computing the fvd metrics on the pretrained digan models
import os
import argparse
from glob import glob 
from tqdm import tqdm
import os.path as osp

digan_path = '/ssd_scratch/cvit/aditya1/digan'
HOME_DIR = '/home2/aditya1/cvit/slp/fvd_results/digan'

# method to extract the fvd score from the text
def get_fvd(result):
    fvd_score = result.split(' ')[1]

    return fvd_score

def compute_fvd(gpu_id, network_pkl, data_path, n_trials=5, num_videos=2048):
    command_format = 'CUDA_VISIBLE_DEVICES={} python src/scripts/compute_fvd_kvd.py \
        --network_pkl {} \
        --data_path {} \
        --n_trials {} \
        --num_videos {}'

    command = command_format.format(gpu_id, network_pkl, data_path, n_trials, num_videos)

    print(f'Command executed : {command}')

    output = os.popen(command).read()
    print(f'The output is : {output}')

    fvd_score = output.split(' ')[2]

    # do something with the output -- send the output directly
    return fvd_score

def save_file(data, filepath, mode, command=None):
    with open(filepath, mode) as f:
        for key, value in data.items():
            f.write(key + '\t' + str(value) + '\n')

        if command is not None:
            f.write(command + '\n')

    
    # with open(filepath, mode) as f:
    #     for item in data:
    #         f.write(item + '\n')

    #     if command is not None:
    #         f.write(command + '\n')


def compute_fvd_dirs(gpu_id, network_dir, data_path, n_trials, n_videos, dataset, result_file, start_from_checkpoint, debug=False, skip_checkpoints=2):
    os.chdir(digan_path)

    # get all the checkpoints from the network_dir
    checkpoints = sorted(glob(network_dir + '/*.pkl'))

    if debug:
        checkpoints = [checkpoints[0]]

    print(f'Original checkpoints : {len(checkpoints)}')

    if start_from_checkpoint > 0:
        print(f'Starting from checkpoint : {start_from_checkpoint}')
        checkpoints = checkpoints[start_from_checkpoint*skip_checkpoints:]

    print(f'Number of checkpoints : {len(checkpoints)}')

    target_dir_path = osp.join(HOME_DIR, dataset)
    os.makedirs(target_dir_path, exist_ok=True)

    results = dict()

    intermediate_file = osp.join(target_dir_path, osp.basename(result_file).split('.')[0] + '_intermediate.txt')
    result_filepath = osp.join(target_dir_path, result_file)

    # skip the checkpoints 
    should_skip = True

    if should_skip:
        updated_checkpoints = [checkpoints[i] for i in range(len(checkpoints)) if i%skip_checkpoints == 0]
        checkpoints = updated_checkpoints

    print(f'After skipping : {len(checkpoints)}')

    for checkpoint in tqdm(checkpoints):
        fvd_score = compute_fvd(gpu_id, checkpoint, data_path, n_trials, n_videos)

        # write the fvd score to the intermediate file 

        save_file({checkpoint : fvd_score}, intermediate_file, 'a')
        # save_file(checkpoint + '\t' + fvd_score, intermediate_file, 'a')

        results[checkpoint] = fvd_score

    # write the complete fvd score to the results file
    save_file(results, result_filepath, 'w', str(network_dir) + '\t' + data_path)


def main(args):
    checkpoint = args.network_pkl
    data_path = args.data_path
    gpu_id = args.gpu_id
    n_trials = args.trials
    videos = args.num_videos
    debug = args.debug
    dataset = args.dataset
    result_file = args.result_file
    start_from_checkpoint = args.start_from

    compute_fvd_dirs(gpu_id, checkpoint, data_path, n_trials, videos, dataset, result_file, start_from_checkpoint, debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_pkl', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--num_videos', type=int, default=2048) # number of videos to evaluate on 
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--start_from', type=int, default=0)

    args = parser.parse_args()
    main(args)