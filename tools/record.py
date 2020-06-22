import time
from mmcv import Config, mkdir_or_exist
import os.path as osp
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Record experimental results')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

def exp_record(cfg_file, record_file = 'work_dirs/Exp_Records.txt'):
    cfg = Config.fromfile(cfg_file)
    timestamp = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())

    backbone = cfg.model.backbone.type + str(cfg.model.backbone.get('depth',''))
    neck = cfg.model.neck.type if cfg.model.get('neck') else ''
    head = cfg.model.head.type[:-4]

    dataset = cfg.dataset_type
    img_scale = cfg.data.train.dataset.img_scale[0]

    partial = 'True' if cfg.data.train.dataset.get('partial_file') is not None else 'False'
    drop_ratio = cfg.data.train.dataset.get('drop_ratio',0)

    work_dir = cfg.work_dir
    if cfg.pseudo:
        pseudo = 'True'
        ps_cfg = cfg.get('pseudo_cfg',None)
        ps_eval_file = osp.join(work_dir, 'temp_ps_eval.txt')
        with open(ps_eval_file, 'r') as f:
            ps_eval = f.read()
    else:
        pseudo = 'False'
        ps_eval = None


    mAP_file = osp.join(work_dir, 'temp_mAP.txt')
    with open(mAP_file,'r') as f:
        mAP = f.read()

    with open(record_file, 'a') as f:

        model_cfg = 'backbone: %-9s | neck: %-5s | head: %-8s'%(backbone,neck,head)
        data_cfg = 'dataset: %-10s | size: %-5s | partial: %-7s | drop_ratio: %-.5f | pseudo: %-5s'%(dataset,img_scale,partial,drop_ratio,pseudo)
        if cfg.pseudo:
            pseudo_cfg = 'interval: %-9d | start: %-4d | high_thrd: %-.6f | low_thrd: %-.6f'%(ps_cfg.interval,ps_cfg.start,ps_cfg.high_thrd,ps_cfg.low_thrd)
        else:
            pseudo_cfg = ''

        records = \
            'Exp Time: ' + timestamp + \
            '\nModel Cfgs: ' + model_cfg.lower() + \
            '\nData Cfgs : ' + data_cfg.lower() + \
            '\nLabel Ops : ' + pseudo_cfg.lower() + \
            '\nWork Dir  : ' + work_dir.lower() + \
            '\nmAP       : \n' + mAP

        if cfg.pseudo:
            records += '\nps_eval   : \n' + ps_eval

        records += '\n' + '='*50 + '\n'

        f.write(records)

if __name__ == '__main__':
    args = parse_args()
    exp_record(args.config)

