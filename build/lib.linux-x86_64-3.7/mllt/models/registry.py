from mllt.utils import Registry


BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
CLASSIFIERS = Registry('classifier')
