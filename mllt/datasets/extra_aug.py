import mmcv
import numpy as np
from numpy import random


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, labels):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, labels):
        if random.randint(2):
            return img, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        return img, labels


class RandomCrop(object):

    def __init__(self, min_crop_size=0.3):
        # 1: return ori img
        self.min_crop_size = min_crop_size

    def __call__(self, img, labels):
        h, w, c = img.shape
       
        for i in range(50):
            new_w = random.uniform(self.min_crop_size * w, w)
            new_h = random.uniform(self.min_crop_size * h, h)

            # h / w in [0.5, 2]
            if new_h / new_w < 0.5 or new_h / new_w > 2:
                continue

            left = random.uniform(w - new_w)
            top = random.uniform(h - new_h)

            patch = np.array((int(left), int(top), int(left + new_w),
                                int(top + new_h)))

            img = img[patch[1]:patch[3], patch[0]:patch[2]]

            return img, labels
        return img, labels


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))

    def __call__(self, img, labels):
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, labels = transform(img, labels)
        return img, labels
