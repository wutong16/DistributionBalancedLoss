from __future__ import division

import os
import sys
sys.path.append(os.getcwd())
import argparse

from mmcv import Config, mkdir_or_exist
import mmcv
import os.path as osp
from mllt.datasets import build_dataset
from mllt.apis import (train_classifier, init_dist, get_root_logger,
                       set_random_seed)
import torch
from mllt.models import build_classifier
import shutil
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument(
        'config', help='train config file path')
    parser.add_argument(
        '--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    # todo: add validate
    parser.add_argument(
        '--validate', action='store_true', help='if validate when training')
    parser.add_argument(
        '--gpus', type=int, default=1, help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    parser.add_argument(
        '--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    # save config file to work dir
    mkdir_or_exist(cfg.work_dir)
    os.system('cp {} {}'.format(args.config, cfg.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)

    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seedllL to {}'.format(args.seed))
        set_random_seed(args.seed)

    train_dataset = build_dataset(cfg.data.train)

    if cfg.model.get('info_dir') is not None:
        mmcv.dump(dict(class_instance_num = train_dataset.class_instance_num.tolist()), osp.join(cfg.model.info_dir))

    model = build_classifier(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.text,
            CLASSES=train_dataset.CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = train_dataset.CLASSES
    train_classifier(
        model, train_dataset, cfg,
        distributed=distributed, validate=args.validate, logger=logger)

    logger.info(cfg.work_dir)

if __name__ == '__main__':
    main()
