import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

from modules.generator import Generator
from modules.mask_generator import MaskGenerator

import torch
import multiprocessing
from train import train
from reconstruction import reconstruction
from animate import animate
from bg_model import BgRefinementNetwork, BackgroundGenerator

if __name__ == "__main__":
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--bg_not_from_dataset",  help='Train with or without backgrounds models',action='store_true')
    parser.add_argument("--without_bg",  help='Train with or without backgrounds models',action='store_true')
    parser.add_argument("--log_dir", default='log', help="path to log into")

    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))), help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    generator = Generator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    mask_generator = MaskGenerator(**config['model_params']['mask_generator_params'],
                             **config['model_params']['common_params'])

    if torch.cuda.is_available():
        mask_generator.to(opt.device_ids[0])

    if opt.verbose:
        print(mask_generator)

    bg_generator = None
    bg_refiner = None
    if not opt.without_bg:
        bg_generator = BackgroundGenerator(3, 64, 512, 2, 6, 60)
        bg_refiner = BgRefinementNetwork()

        if torch.cuda.is_available():
            bg_generator.to(opt.device_ids[0])
            bg_refiner.to(opt.device_ids[0])

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, mask_generator, bg_generator,bg_refiner,  opt.checkpoint, log_dir, dataset, opt.device_ids, opt.without_bg)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, mask_generator, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        multiprocessing.set_start_method('spawn', True)
        print("Animate...")
        animate(config, generator, mask_generator,bg_generator, bg_refiner, opt.checkpoint, log_dir, dataset, opt.bg_not_from_dataset)
    # elif opt.mode == 'animate_w_bg':
    #     multiprocessing.set_start_method('spawn', True)
    #     print("Animate...")
    #     animate_w_bg(config, generator, mask_generator, opt.checkpoint, log_dir, dataset, )
