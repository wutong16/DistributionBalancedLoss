from mllt.datasets.datasets_split import LVIS_UNSEEN_ID, LVIS_SEEN_ID
# model settings
model = dict(
    type='SimpleClassifier',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='PFC',
        in_channels=2048,
        out_channels=1024,
        dropout=0.5),
    head=dict(
        type='ClsHead',
        in_channels=1024,
        num_classes=830,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict()
test_cfg = dict()

# dataset settings
dataset_type = 'LvisDataset'
data_root = '/mnt/SSD/det/coco/'
online_data_root = 'appendix/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
extra_aug = dict(
    photo_metric_distortion=dict(
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    random_crop=dict(
        min_crop_size=0.8
    )
)
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file= 'data/LVIS/lvis_v0.5_train.json',
        img_prefix=data_root + 'train2017/',
        LT_ann_file = [online_data_root + 'lvis/longtail/img_id.pkl'],
        see_only=LVIS_SEEN_ID,
        img_scale=(224, 224),
        img_norm_cfg=img_norm_cfg,
        extra_aug=extra_aug,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0.5),
    val=dict(
        type=dataset_type,
        ann_file= 'data/LVIS/lvis_v0.5_val.json',
        img_prefix=data_root + 'val2017/',
        LT_ann_file = [online_data_root + 'lvis/longtail/img_id.pkl'],
        see_only=LVIS_SEEN_ID,
        img_scale=(224, 224),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0),
    test=dict(
        type=dataset_type,
        ann_file= 'data/LVIS/lvis_v0.5_val.json',
        img_prefix=data_root + 'val2017/',
        class_split=online_data_root + 'lvis/longtail/class_split.pkl',
        see_only=LVIS_SEEN_ID,
        img_scale=(224, 224),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001,
                 # paramwise_options=dict(
                 #     param_group_cfg=dict(
                 #         backbone=0.05,
                 #         head=2
                 #     ))
                 )
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 3,
    step=[55, 70])
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=1)
# runtime settings
total_epochs = 80
dist_params = dict(backend='nccl')
import logging
log_level = logging.INFO #'INFO'
work_dir = './work_dirs/lvis_resnet50_pfc_830_b8'
load_from = None
resume_from = None
workflow = [('train', 1)]
