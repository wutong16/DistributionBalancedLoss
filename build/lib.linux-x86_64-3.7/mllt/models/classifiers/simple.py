import torch.nn as nn

from .base import BaseClassifier
from .. import builder
from ..registry import CLASSIFIERS
from mmcv.parallel import DataContainer as DC
import torch
import numpy as np

@CLASSIFIERS.register_module
class SimpleClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 head,
                 neck=None,
                 lock_back=False,
                 lock_neck=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 savefeat=False):
        super(SimpleClassifier, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.lock_back=lock_back
        self.lock_neck=lock_neck
        self.savefeat=savefeat
        if self.savefeat and not self.with_neck:
            assert neck is not None, 'We must have a neck'
            assert train_cfg is None, 'this is only at testing stage'
        if self.lock_back:
            print('\033[1;35m >>> backbone locked !\033[0;0m')
        if self.lock_neck:
            print('\033[1;35m >>> neck locked !\033[0;0m')
        self.init_weights(pretrained=pretrained)
        self.count = CountMeter(num_classes=20)

        # freq = torch.tensor([5,6,6,5,7,16,12,5,9,6,5,5,10,21,6,5],dtype=torch.float)
        # self.cla_weight = torch.mean(torch.sqrt(freq))*(torch.ones(freq.shape,dtype=torch.float) / torch.sqrt(freq)).cuda()
        # print(self.cla_weight)


    def init_weights(self, pretrained=None):
        super(SimpleClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.head.init_weights()

    def extract_feat(self, img):
        if self.lock_back:
            with torch.no_grad():
                x = self.backbone(img)
        else:
            x = self.backbone(img)

        if self.with_neck:
            if self.lock_neck:
                with torch.no_grad():
                    x = self.neck(x)
            else:
                x = self.neck(x)

        return x

    def forward_train(self,
                      img,
                      img_metas, 
                      gt_labels):
        # if self.lock_back:
        #     with torch.no_grad():
        #         x = self.extract_feat(img)
        # else:
        #     x = self.extract_feat(img)
        x = self.extract_feat(img)
        outs = self.head(x)

        loss_inputs = (outs, gt_labels)
        losses = self.head.loss(*loss_inputs)

        # self.count.update(gt_labels)

        return losses

    # def simple_test(self, img, img_meta=None, rescale=False):
    #     x = self.extract_feat(img)
    #     outs = self.head(x)
    #     return outs

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.head(x)

        if self.savefeat:
            return outs, x

        return outs

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

''' This part is only for resampling
'''
class CountMeter(object):
    def __init__(self, num_classes, vector_len=1, end_n=15500): # coco reduce 4: 22560 # voc reduce 1: 15500
        self.num_classes = num_classes
        self.end_n = end_n
        self.vector_len = vector_len
        self.reset()

    def reset(self):
        self.count = torch.zeros(self.num_classes, dtype=torch.int64).cuda()
        self.all_features = None
        self.all_labels = None
        self.n = 0

    def update(self, gt_labels, features=None):
        n = gt_labels.shape[0]
        self.count += torch.sum(gt_labels, dim=0)
        if self.all_labels is None:
            self.all_labels = gt_labels.cpu().numpy()
            self.all_features = features.cpu().numpy()
        else:
            self.all_labels = np.concatenate((self.all_labels, gt_labels.cpu().numpy()))
            self.all_features = np.concatenate((self.all_features, features.cpu().numpy()))
        self.n += n
        if self.n >= self.end_n:
            self.save_and_exit()

    def save_and_exit(self):
        import mmcv
        data = dict(count=self.count.cpu().numpy(), all_labels=self.all_labels)
        mmcv.dump(data, './mllt/appendix/VOCdevkit/longtail2012/resample_results_b6.pkl')
        print('resample result saved at :{}'.format('./mllt/appendix/VOCdevkit/longtail2012/resample_results_b6.pkl'))
        exit()


