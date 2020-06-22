# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
from .custom import CustomDataset
from .registry import DATASETS
import mmcv
import random
import pickle

category_file = './data/miniImagenet/categories.pkl'
@DATASETS.register_module
class miniImagenetDataset(CustomDataset):

    if os.path.exists(category_file):
        with open(category_file,'rb') as f:
            CLASSES = pickle.load(f)['CLASSES']
    else:
        CLASSES = range(80)

    # def __getitem__(self, i):
    #     image_path = os.path.join(self.meta['image_names'][i])
    #     img = Image.open(image_path).convert('RGB')
    #     target = self.target_transform(self.meta['image_labels'][i])
    #     return img, target

    def __len__(self):
        return len(self.meta['image_names'])

    def load_annotations(self, ann_file, LT_ann_file=None):
        ann_file = LT_ann_file if LT_ann_file is not None else ann_file
        if isinstance(ann_file, list):
            ann_file = ann_file[0]
        with open(ann_file, 'r') as f:
            self.meta = json.load(f)
            print('Loading annotation from {}'.format(ann_file))
        self.target_transform = lambda x: x
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.CLASSES)
        }

        self.img_ids = [i for i in range(len(self.meta['image_names']))]
        img_infos = []
        for i in self.img_ids:
            info = dict(filename=os.path.join(self.meta['image_names'][i]))
            info.update(id=i, width=224, height=224)

            img_infos.append(info)

        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        target = self.target_transform(self.meta['image_labels'][idx])
        category_name = self.meta['image_names'][idx].split('/')[-2]
        target = self.cat2label[category_name]
        ann = self._parse_ann_info(label_idx=target)
        return ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, label_idx):
        """Parse label annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
        Returns:
            dict: A dict containing the following key: labels
        """
        if self.single_label:
            ann = dict(labels=label_idx)
        else:
            gt_labels = np.zeros((len(self.CLASSES), ), dtype=np.int64)
            gt_labels[label_idx] = 1
            ann = dict(labels=gt_labels)
        return ann
