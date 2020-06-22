import numpy as np
from mllt.datasets.dataset_wrappers import ConcatDataset, RepeatDataset
import pickle
from mllt.datasets import build_dataset
import os
import os.path as osp
from mmcv import Config, mkdir_or_exist
from .display import *
import shutil
import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import time

def eval_long_tail(train_results, train_gt_labels, results, gt_labels, save_dir='./', name=None):
    train_gt_labels, gt_labels, results = lists_to_arrays([train_gt_labels, gt_labels, results])
    num_classes = gt_labels.shape[1]
    sample_num = np.sum(train_gt_labels, axis=0)
    test_sample_num = np.sum(gt_labels, axis=0)
    rank_idx = np.argsort(-sample_num)
    APs = average_precision_score(gt_labels, results, None)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title('train time')
    ''' 
    # sample number
    ax1.plot(range(num_classes), sample_num[rank_idx], alpha=0.4, linestyle="-.")
    if real_train_data is not None:
        real_train_count = real_train_data['count']
        ax4 = ax1.twinx()
        ax4.plot(range(num_classes), real_train_count[rank_idx], alpha=0.4, color='purple', label='real')
        ax4.legend()
    ax1.legend()
    '''
    ax2 = fig.add_subplot(2, 1, 2)
    if train_results is not None:
        train_avg_pos = np.sum(train_results * train_gt_labels, axis=0) / np.sum(train_gt_labels, axis=0)
        train_avg_neg = np.sum(train_results * (1 - train_gt_labels), axis=0) / np.sum(1 - gt_labels, axis=0)
        ax1.plot(range(num_classes), train_avg_pos[rank_idx], color='green', alpha=0.8, label='avg_pos')
        ax1.plot(range(num_classes), train_avg_neg[rank_idx], color='red',alpha=0.8, label='avg_neg')
        #
    avg_pos = np.sum(results * gt_labels, axis=0) / np.sum(gt_labels, axis=0)
    avg_neg = np.sum(results * (1 - gt_labels), axis=0) / np.sum(1 - gt_labels, axis=0)


    ax2.set_title('test time')
    ax2.plot(range(num_classes), avg_pos[rank_idx], color='green', alpha=0.8, label='avg_pos') # -{:.2f}'.format(pos_move_up))
    ax2.plot(range(num_classes), avg_neg[rank_idx], color='red', alpha=0.8, label='avg_neg') # -{:.2f}'.format(neg_move_down))
    ax2.legend()

    ''' # mAP
    ax3 = ax2.twinx()
    ax3.bar(range(num_classes), APs[rank_idx], alpha=0.5, label='mAP')
    ax3.legend()
    '''

    name = 'eval_lt.jpg' if name is None else name
    plt.savefig(osp.join(save_dir, name))
    print('Long tail results saved at {}'.format(osp.join(save_dir, name)))

    index_per_class = []
    co_occur = []
    for cla in range(num_classes):
        indexs = np.where(train_gt_labels[:, cla] > 0)[0].tolist()
        index_per_class.append(indexs)
        co_occur.append(np.sum(train_gt_labels[indexs], axis=0))

    less_than = 20
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    for cla in range(num_classes):
        x = []
        y = []
        if sample_num[cla] >= less_than or APs[cla] > 0.2:
            continue
        min_pos = np.min((gt_labels * results)[:,cla])
        bad_idx = np.where(((1 - gt_labels) * results)[:,cla] > min_pos)[0]
        # print(cla, sample_num[cla], 'AP {:.3f}, min_pos {:.3f} in {} Pos, {} bad in {} Neg'.format(
        #     APs[cla], min_pos, np.sum(gt_labels[:,cla]), len(bad_idx), np.sum(( 1 - gt_labels)[:,cla])))
        for i in range(num_classes):
            if co_occur[cla][i] > 0 and i != cla:
                # sus_neg = np.mean(gt_labels[:, i] * (1 - gt_labels)[:, cla] * results[:,cla])
                # sus_pos = np.mean(gt_labels[:, i] * gt_labels[:, cla] * results[:, cla])
                sus_neg = _mean_top_k(gt_labels[:, i] * (1 - gt_labels)[:, cla] * results[:,cla], k=100)
                sus_pos = _mean_top_k(gt_labels[:, i] * gt_labels[:, cla] * results[:, cla], k=100)
                x.append(co_occur[cla][i] / sample_num[cla])
                y.append([sus_neg, sus_pos])
                # print('{:.2f}/{:.2f}/{:.3f}/{:.3f}'.format(co_occur[cla][i] / sample_num[cla] , sample_num[i] / sample_num[cla], sus_neg, sus_pos), end=' ')
            # else:
            #     print('   ---   ', end=' ')
        # print('')
        y = np.asarray(y)
        x = np.asarray(x)
        idx = np.argsort(x)
        ax.plot(x[idx], y[:, 0][idx])
        ax1.plot(x[idx], y[:, 1][idx])
    # plt.legend()
    plt.savefig(osp.join(save_dir, 'co_pos_neg.jpg'))

def eval_F1(results, gt_labels):
    y_pred = np.asarray(results) > 0
    micro_f1 = f1_score(gt_labels, y_pred, average='micro')
    macro_f1 = f1_score(gt_labels, y_pred, average='macro')
    return micro_f1, macro_f1

def eval_acc(results, gt_labels):
    y_pred = np.asarray(results) > 0
    tp = gt_labels * y_pred
    tn = (1 - gt_labels) * (1 - y_pred)
    fp = (1 - gt_labels) * y_pred
    fn = gt_labels * (1 - y_pred)
    assert np.sum(tp + tn + fp + fn) == np.shape(gt_labels)[0]*np.shape(gt_labels)[1]
    per_class_acc = np.sum(tp + tn, axis=0) / np.sum(tp + tn + fp + fn, axis=0)
    acc = np.mean(per_class_acc)
    return acc, per_class_acc

def _non_zero_mean(data, axis=0):
    data_sum = np.sum(data, axis=axis)
    data_num = np.sum(data != 0, axis=axis)
    return data_sum / data_num

def _non_zero_var(data, axis=0):
    data_sum = np.sum(data, axis=axis)
    mean = _non_zero_mean(data)
    square_sum = np.sum((data - mean*(data != 0)) ** 2, axis=axis)
    var = square_sum / data_sum
    return var

def _mean_top_k(x, k=3, nonzero=True):
    if nonzero:
        x = x[np.where(x != 0)[0]]

    if k > len(x):
        return np.mean(x)
    sorted = - np.sort(- x)
    return np.mean(sorted[:k])

def sigmoid(x):
    if isinstance(x,list):
        x = np.asarray(x)
    return 1 / (1 + np.exp(-x))

def lists_to_arrays(lists):
    arrays = []
    for obj in lists:
        if obj is None:
            obj = None
        elif isinstance(obj,list):
            obj = np.asarray(obj)
        elif not isinstance(obj,np.ndarray):
            raise Exception('object not list or ndarray!')
        arrays.append(obj)
    return arrays


