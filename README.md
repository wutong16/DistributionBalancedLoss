# Distribution Balance Loss

Code for our paper *Distribution-Balanced Loss for Multi-Label
Classification in Long-Tailed Datasets*,  submitted to ECCV2020.

## TODO
- [ ] Clean the Repo
- [ ] Distributed training is not supported currently
- [ ] Evaluation with single GPU is not supported currently
- [ ] uncertainty control of `create_longtail_dataset.py`
- [ ] Running in the new environment is a little bit worse than in the original one, currently trying to fix the problem

## Requirements
* Pytorch 1.1.0
* Sklearn

## Datasets
Our long-tail multi-label datasets are created from MS COCO 2017 and Pascal VOC 2012. Annotations and statistics data resuired when training are saved under `./appendix`
You can create a new long-tail dataset with the following command. 
```
python tools/create_longtail_dataset.py
```

## Run training
#### Installation
```
git clone git@github.com:wutong16/DistributionBalance.git
cd DistributionBalance
```
#### Train with COCO-MLT
```
python tools/train.py configs/coco/LT_resnet50_pfc_DB.py 
```
#### Train with VOC-MLT
```
python tools/train.py configs/voc/LT_resnet50_pfc_DB.py 
```

## Evaluation

#### COCO-MLT

```
bash tools/dist_test.sh configs/coco/LT_resnet50_pfc_DB.py work_dirs/LT_coco_resnet50_pfc_DB/epoch_8.pth 1
```
#### VOC-MLT

```
bash tools/dist_test.sh configs/voc/LT_resnet50_pfc_DB.py work_dirs/LT_voc_resnet50_pfc_DB/epoch_8.pth 1
```

## Contact

This repo is currently maintained by Tong Wu ([@wutong16](https://github.com/wutong16)) and Qingqiu Huang ([@HQQ](https://github.com/hqqasw))
