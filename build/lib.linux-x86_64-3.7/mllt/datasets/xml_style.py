import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class XMLDataset(CustomDataset):

    def __init__(self, **kwargs):
        super(XMLDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.class_instance_num = self.get_class_instance_num()
        self.index_dic = self.get_index_dic()

    def load_annotations(self, ann_file, LT_ann_file=None):
        img_infos = []
        self.img_ids = mmcv.list_from_file(ann_file)
        for img_id in self.img_ids:
            filename = 'JPEGImages/{}.jpg'.format(img_id)
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos

    def get_ann_info(self, idx, epoch=0):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations',
                            '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        gt_labels = np.zeros((len(self.CLASSES), ), dtype=np.int64)
        for obj in root.findall('object'):
            name = obj.find('name').text
            gt_labels[self.cat2label[name]-1] = 1
        ann = dict(labels=gt_labels)
        return ann

