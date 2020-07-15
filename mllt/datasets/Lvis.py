import numpy as np
from lvis.lvis import LVIS
from .custom import CustomDataset
from .registry import DATASETS
import os.path as osp
import mmcv


@DATASETS.register_module
class LvisDataset(CustomDataset):

    def load_annotations(self, ann_file, LT_ann_file=None):
        self.lvis = LVIS(ann_file)
        self.cat_ids = self.lvis.get_cat_ids()
        if self.CLASSES is None:
            self.CLASSES = self.cat_ids
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.categories = self.cat_ids

        if LT_ann_file is not None:
            self.img_ids = []
            for LT_ann_file in LT_ann_file:
                self.img_ids += mmcv.list_from_file(LT_ann_file)
                self.img_ids = [ int(x) for x in self.img_ids]
        else:
            self.img_ids = self.lvis.get_img_ids()

        img_infos = []
        for i in self.img_ids:
            info = self.lvis.load_imgs([i])[0]
            info['filename'] = info['file_name'][-16:]
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        ann_info = self.lvis.load_anns(ann_ids)
        ann = self._parse_ann_info(ann_info)
        if self.see_only:
            ann['labels'] = ann['labels'][list(self.see_only)]
        return ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.lvis.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info):
        """Parse label annotation. """
        gt_labels = np.zeros((len(self.categories),), dtype=np.int64)
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            gt_labels[self.cat2label[ann['category_id']]-1] = 1

        ann = dict(labels=gt_labels)

        return ann