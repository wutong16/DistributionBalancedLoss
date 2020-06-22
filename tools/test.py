import argparse
import os
import tempfile
import os.path as osp
import shutil
import numpy as np
import resource
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import sys
sys.path.append(os.getcwd())

from mllt.datasets.dataset_wrappers import ConcatDataset, RepeatDataset
from mllt.datasets import build_dataset
from mllt.apis import init_dist
from mllt.datasets import build_dataloader
from mllt.models import build_classifier
from mllt.core.evaluation.eval_tools import lists_to_arrays, eval_acc, eval_F1
from mllt.core.evaluation.mean_ap import eval_map
from mllt.models.losses import accuracy

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset),bar_width=20)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result.cpu().numpy())
        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset),bar_width=20)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result.cpu().numpy())

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results

def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument(
        'config', help='test config file path')
    parser.add_argument(
        'checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out', help='output result file')
    parser.add_argument(
        '--eval', type=str, nargs='+', choices=['mAP', 'multiple'],
        default=['multiple'], help='eval metrics')
    parser.add_argument(
        '--show', default=True, help='show results')
    parser.add_argument(
        '--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    parser.add_argument(
        '--local_rank', type=int, default=0)
    parser.add_argument(
        '--testset_only', type=bool, default=True, help='only eval test set')
    parser.add_argument(
        '--from_file', action='store_true', help='load network output results')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

# make a dataset list with both test and training set
# the training set has to maintain its get_ann_info() properties
def make_dataset_list(cfg, test_only=True):
    cfg.data.test.test_mode = True
    test_dataset = build_dataset(cfg.data.test)
    dataset_list = [test_dataset]
    if test_only:
        return dataset_list

    if cfg.data.train.get('dataset',None) is not None:
        train_cfg = cfg.data.train.dataset
    else:
        train_cfg = cfg.data.train
    train_cfg.test_mode = True
    train_cfg.extra_aug = None
    train_cfg.flip_ratio = 0
    train_dataset = build_dataset(train_cfg)

    if isinstance(train_dataset, ConcatDataset):
        train_datasets = train_dataset.datasets
    else:
        train_datasets = [train_dataset]

    for train_dataset in train_datasets:
        dataset_list.append(train_dataset)

    return dataset_list

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.work_dir = osp.abspath(osp.join(args.checkpoint, '..'))

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    if args.out is None:
        epoch = args.checkpoint.split('.')[-2].split('_')[-1]
        args.out = osp.join(cfg.work_dir, 'gt_and_results_e{}.pkl'.format(epoch))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    rank, _ = get_dist_info()

    if args.from_file:
        savedata = mmcv.load(args.out)

    else:
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None

        # build the dataloader
        dataset_list = make_dataset_list(cfg, args.testset_only)
        # build the model and load checkpoint
        model = build_classifier(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

        if osp.exists(args.out):
            savedata = mmcv.load(args.out)
        else:
            savedata = [dict() for _ in range(len(dataset_list))]
        for d, dataset in enumerate(dataset_list):
            if args.testset_only and d > 0:
                break
            data_loader = build_dataloader(
                dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)

            gt_labels = []
            for i in range(len(dataset)):
                gt_ann = dataset.get_ann_info(i)
                gt_labels.append(gt_ann['labels'])

            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = dataset.CLASSES

            if not distributed:
                model = MMDataParallel(model, device_ids=[0])
                outputs = single_gpu_test(model, data_loader, args.show)
            else:
                model = MMDistributedDataParallel(model.cuda())
                outputs = multi_gpu_test(model, data_loader, args.tmpdir)

            if rank == 0:
                savedata[d].update(gt_labels=gt_labels,outputs=np.vstack(outputs))
        if rank == 0:
            print('\nwriting results to {}'.format(args.out))
            mmcv.dump(savedata, args.out)

    img_prefixs = []
    dataset_list = make_dataset_list(cfg, args.testset_only)
    img_ids = [[] for _ in range(len(dataset_list))]
    for d, dataset in enumerate(dataset_list):
        img_prefixs.append(dataset.img_prefix)
        img_infos = dataset.img_infos
        for i in range(len(dataset)):
            img_ids[d].append(img_infos[i]['id'])

    if rank == 0:
        display_dict = {}
        eval_metrics = args.eval
        dataset = build_dataset(cfg.data.test)
        display_dict['class'] = dataset.CLASSES
        for i, data in enumerate(savedata):
            if args.testset_only and i > 0: # test-set
                break
            gt_labels = data['gt_labels']
            outputs = data['outputs']

            gt_labels, outputs = lists_to_arrays([gt_labels, outputs])
            print('Starting evaluate {}'.format(' and '.join(eval_metrics)))
            for eval_metric in eval_metrics:
                if eval_metric == 'mAP':
                    mAP, APs = eval_map(outputs, gt_labels, None, print_summary=True)
                    display_dict['APs_{:1d}'.format(i)] = APs
                elif eval_metric == 'multiple':
                    metrics = []
                    for split, selected in dataset.class_split.items():
                        selected = list(selected)
                        selected_outputs = outputs[:, selected]
                        selected_gt_labels = gt_labels[:, selected]
                        classes = np.asarray(dataset.CLASSES)[selected]
                        mAP, APs = eval_map(selected_outputs, selected_gt_labels, classes, print_summary=False)
                        micro_f1, macro_f1 = eval_F1(selected_outputs, selected_gt_labels)
                        acc, per_cls_acc = eval_acc(selected_outputs, selected_gt_labels)
                        metrics.append([split, mAP, micro_f1, macro_f1, acc])
                    mAP, APs = eval_map(outputs, gt_labels, dataset, print_summary=False)
                    micro_f1, macro_f1 = eval_F1(outputs, gt_labels)
                    acc, per_cls_acc = eval_acc(outputs, gt_labels)
                    metrics.append(['Total', mAP, micro_f1, macro_f1, acc])
                    for split, mAP, micro_f1, macro_f1, acc in metrics:
                        print('Split:{:>6s} mAP:{:.4f}  acc:{:.4f}  micro:{:.4f}  macro:{:.4f}'.format(
                            split, mAP, acc, micro_f1, macro_f1))

if __name__ == '__main__':
    main()
