import copy

from mllt.utils import build_from_cfg
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .registry import DATASETS


def _concat_dataset(cfg):
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    partial_files = cfg.get('partial_file', None)
    pseudo_files = cfg.get('pseudo_file', None)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(partial_files, (list, tuple)):
            data_cfg['partial_file'] = partial_files[i]
        if isinstance(pseudo_files, (list, tuple)):
            data_cfg['pseudo_file'] = pseudo_files[i]
        datasets.append(build_dataset(data_cfg))

    return ConcatDataset(datasets)


def build_dataset(cfg):
    if cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(build_dataset(cfg['dataset']), cfg['times'])
    elif isinstance(cfg['ann_file'], (list, tuple)):
        dataset = _concat_dataset(cfg)
    else:
        dataset = build_from_cfg(cfg, DATASETS)

    return dataset
