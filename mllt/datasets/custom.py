import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .registry import DATASETS
from .transforms import ImageTransform, Numpy2Tensor
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation
import cv2

@DATASETS.register_module
class CustomDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'labels': <np.ndarray> (n, )
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 LT_ann_file=None,
                 multiscale_mode='value',
                 size_divisor=None,
                 flip_ratio=0,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 class_split=None,
                 see_only=set(),
                 save_info=False):

        # prefix of images path
        self.img_prefix = img_prefix
        # self.single_label = True if 'Imagenet' in ann_file else False
        self.single_label = False

        # load annotations (and proposals)
        if LT_ann_file is not None:
            self.img_infos = self.load_annotations(ann_file, LT_ann_file)
        else:
            self.img_infos = self.load_annotations(ann_file)
        self.ann_file = ann_file
        self.see_only = see_only
        # filter images with no annotation during training
        if not test_mode and 'width' in self.img_infos[0].keys():
            min_size = 32
            valid_inds = []
            for i, img_info in enumerate(self.img_infos):
                if min(img_info['width'], img_info['height']) >= min_size:
                    valid_inds.append(i)
            # valid_inds = self._filter_imgs()

            self.img_infos = [self.img_infos[i] for i in valid_inds]
        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

        if class_split is not None:
            self.class_split = mmcv.load(class_split)

        self.save_info = save_info

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file, LT_ann_file=None):
        return mmcv.load(ann_file)

    def get_index_dic(self, list=False, get_labels=False):
        """ build a dict with class as key and img_ids as values
        :return: dict()
        """
        if self.single_label:
            return

        num_classes = len(self.get_ann_info(0)['labels'])
        gt_labels = []
        idx2img_id = []
        img_id2idx = dict()
        co_labels = [[] for _ in range(num_classes)]
        condition_prob = np.zeros([num_classes, num_classes])

        if list:
            index_dic = [[] for i in range(num_classes)]
        else:
            index_dic = dict()
            for i in range(num_classes):
                index_dic[i] = []

        for i, img_info in enumerate(self.img_infos):
            img_id = img_info['id']
            label = self.get_ann_info(i)['labels']
            gt_labels.append(label)
            idx2img_id.append(img_id)
            img_id2idx[img_id] = i
            for idx in np.where(np.asarray(label) == 1)[0]:
                index_dic[idx].append(i)
                co_labels[idx].append(label)

        for cla in range(num_classes):
            cls_labels = co_labels[cla]
            num = len(cls_labels)
            condition_prob[cla] = np.sum(np.asarray(cls_labels), axis=0) / num

        ''' save original dataset statistics, run once!'''
        if self.save_info:
            self._save_info(gt_labels, img_id2idx, idx2img_id, condition_prob)

        if get_labels:
            return index_dic, co_labels
        else:
            return index_dic

    def get_class_instance_num(self):
        gt_labels = []
        for i, img_info in enumerate(self.img_infos):
            ann = self.get_ann_info(i)['labels']
            gt_labels.append(ann)
        class_instance_num = np.sum(gt_labels, 0)
        return class_instance_num

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            if 'width' not in self.img_infos[0].keys():
                self.flag[i] = i % 2
                continue
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
            elif img_info['width'] / img_info['height'] == 1:
                self.flag[i] = i % 2

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))

        ann = self.get_ann_info(idx)
        gt_labels = ann['labels']

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_labels = self.extra_aug(img, gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_labels=to_tensor(gt_labels))
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))

        def prepare_single(img, scale, flip):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            return _img, _img_meta

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta = prepare_single(img, scale, False)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            if self.flip_ratio > 0:
                _img, _img_meta = prepare_single(img, scale, True)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))

        data = dict(img=imgs, img_meta=img_metas)
        return data

    def prepare_raw_img(self, idx):
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))

        _img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, self.img_scales[0], flip=False, keep_ratio=self.resize_keep_ratio)
        img_meta = dict(
            ori_shape=(img_info['height'], img_info['width'], 3),
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)

        data = dict(img=img, img_meta=img_meta)
        return data

    def _save_info(self, gt_labels, img_id2idx, idx2img_id, condition_prob):
        '''save info for later training'''
        ''' save original gt_labels '''
        save_data = dict(gt_labels=gt_labels, img_id2idx=img_id2idx, idx2img_id=idx2img_id)
        if 'coco' in self.ann_file:
            # path = 'mllt/appendix/coco/terse_gt_2017_test.pkl'
            path = 'mllt/appendix/coco/terse_gt_2017.pkl'
        elif 'VOC' in self.ann_file:
            path = 'mllt/appendix/VOCdevkit/terse_gt_2012.pkl'
            # path = 'mllt/appendix/VOCdevkit/terse_gt_2007_test.pkl'
        else:
            raise NameError

        if not osp.exists(path):
            mmcv.dump(save_data, path)
            print('key info saved at {}!'.format(path))
        else:
            print('already exist, wont\'t overwrite!')

        ''' save long tail information '''
        class_freq = np.sum(gt_labels, axis=0)
        # print(np.mean(class_freq), np.var(class_freq/len(gt_labels)))
        neg_class_freq = np.shape(gt_labels)[0] - class_freq
        save_data = dict(gt_labels=gt_labels, class_freq=class_freq, neg_class_freq=neg_class_freq
                         , condition_prob=condition_prob)
        if 'coco' in self.ann_file:
            # long-tail coco
            path = 'mllt/appendix/coco/longtail2017/class_freq.pkl'
            # full coco
            # path = 'mllt/appendix/coco/class_freq.pkl'
        elif 'VOC' in self.ann_file:
            # long-tail VOC
            path = 'mllt/appendix/VOCdevkit/longtail2012/class_freq.pkl'
            # full VOC
            # path = 'mllt/appendix/VOCdevkit/class_freq.pkl'
        else:
            raise NameError

        if not osp.exists(path):
            mmcv.dump(save_data, path)
            print('key info saved at {}!'.format(path))
        else:
            print('already exist, wont\'t overwrite!')
        exit()