import mmcv
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
COCO_SEEN = set()
COCO_UNSEEN = ('bicycle', 'boat', 'stop_sign', 'bird', 'backpack','frisbee', 'snowboard', 'surfboard',
          'cup', 'fork', 'spoon', 'broccoli', 'chair', 'keyboard', 'microwave', 'vase')
COCO_UNSEEN_ID = set()
COCO_SEEN_ID = set()
for id, cat in enumerate(COCO_CLASSES):
    if cat in COCO_UNSEEN:
        COCO_UNSEEN_ID.add(id)
    else:
        COCO_SEEN_ID.add(id)
        COCO_SEEN.add(cat)

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

lvis_data = mmcv.load('./appendix/lvis/longtail/class_data.pkl')
LVIS_CLASSES = lvis_data['LVIS_CLASSES']
LVIS_SEEN = lvis_data['LVIS_SEEN']
LVIS_UNSEEN = lvis_data['LVIS_UNSEEN']

LVIS_SEEN_ID = set()
LVIS_UNSEEN_ID = set()
for id, cat in enumerate(LVIS_CLASSES):
    if cat in LVIS_SEEN:
        LVIS_SEEN_ID.add(id)
    else:
        assert cat in LVIS_UNSEEN
        LVIS_UNSEEN_ID.add(id)
