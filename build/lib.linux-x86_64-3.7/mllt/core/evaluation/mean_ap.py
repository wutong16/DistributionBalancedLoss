import mmcv
import numpy as np
from terminaltables import AsciiTable

from .class_names import get_classes
from sklearn.metrics import average_precision_score, accuracy_score
import torch

def eval_map(results,
             gt_labels,
             dataset=None,
             print_summary=True):
    """Evaluate mAP of a dataset.

    Args:
        results (ndarray): shape (num_samples, num_classes)
        gt_labels (ndarray): ground truth labels of each image, shape (num_samples, num_classes)
        dataset (None or str or list): dataset name or dataset classes, there
            are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc.
        print_summary (bool): whether to print the mAP summary

    Returns:
        tuple: (mAP, [AP_1, AP_2, ..., AP_C]
    """
    results = np.asarray(results)
    gt_labels = np.asarray(gt_labels)
    assert results.shape[0] == gt_labels.shape[0]
    eval_results = []
    num_samples, num_classes = results.shape
    # print(results)
    APs = average_precision_score(gt_labels, results, None)
    mAP = APs.mean()
    if print_summary:
        print_map_summary(mAP, APs, dataset)
    return mAP, APs


def print_map_summary(mAP, APs, dataset=None):
    """Print mAP and results of each class.

    Args:
        mAP(float): calculated from `eval_map`
        APs(ndarray): calculated from `eval_map`
        dataset(None or str or list): dataset name or dataset classes.
    """

    num_classes = APs.shape[0]

    if dataset is None:
        label_names = [str(i) for i in range(1, num_classes + 1)]
    elif mmcv.is_str(dataset):
        label_names = get_classes(dataset)
    else:
        label_names = list(dataset)

    header = ['class', 'ap']

    table_data = [header]
    for j in range(num_classes):
        row_data = [
            label_names[j], '{:.4f}'.format(APs[j])
        ]
        table_data.append(row_data)
    table_data.append(['mAP', '{:.4f}'.format(mAP)])
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print(table.table)


def binary_tfpn(gt_labels, labels, masks, by_class=False):
    '''
    mask : 1 represent where we would look at
    '''
    if isinstance(gt_labels, list):
        gt_labels = np.asarray(gt_labels)
    if isinstance(labels, list):
        labels = np.asarray(labels)
    if isinstance(masks, list):
        masks = np.asarray(masks)
    # mask = 1 - partial_mask
    valid_num = np.sum(masks)

    # print('label',label.tolist())
    # print('gt_label',gt_label.tolist())
    # print('mask',mask.tolist())
    tp_vec = labels * gt_labels * masks
    tn_vec = (1 - labels) * (1 - gt_labels) * masks
    fp_vec = labels * (1 - gt_labels) * masks
    fn_vec = (1 - labels) * gt_labels * masks

    if by_class:
        accuracy = (np.sum(tp_vec,0) + np.sum(tn_vec,0)) / \
                   (np.sum(tp_vec,0) + np.sum(tn_vec,0) + np.sum(fp_vec,0) + np.sum(fn_vec,0))
        p_pre = np.sum(tp_vec,0) / (np.sum(tp_vec,0) + np.sum(fp_vec,0))
        n_pre = np.sum(tn_vec,0) / (np.sum(tn_vec,0) + np.sum(fn_vec,0))
        recall = np.sum(tp_vec,0) / (np.sum(tp_vec,0) + np.sum(fn_vec,0))
    else:
        accuracy = (np.sum(tp_vec) + np.sum(tn_vec)) / (
                    np.sum(tp_vec) + np.sum(tn_vec) + np.sum(fp_vec) + np.sum(fn_vec))
        p_pre = np.sum(tp_vec) / (np.sum(tp_vec) + np.sum(fp_vec))
        n_pre = np.sum(tn_vec) / (np.sum(tn_vec) + np.sum(fn_vec))
        recall = np.sum(tp_vec) / (np.sum(tp_vec) + np.sum(fn_vec))


    return accuracy, p_pre, n_pre, recall


def statistic_category(labels, masks=None):
    if isinstance(labels, list):
        labels = np.asarray(labels)
    if isinstance(masks, list):
        masks = np.asarray(masks)

    if masks is not None:
        labels = labels * masks

    total_instances = np.sum(labels)
    each_category = np.sum(labels, axis=0)
    freq_category = each_category / total_instances
    return freq_category


def display_comparison(Dict, col=5):
    '''
    :param Dict: items to display together  {'namex':[x1,x2,...,xn],'namey':[y1,y2,...,yn]}
    :param col: how many cols in a row
    :return:
    '''
    row = 0
    end = False
    while not end:
        for key, value_list in Dict.items():
            print('%10s:' % key, end='')
            for i in range(col):
                idx = row * col + i
                if isinstance(value_list[idx], float):
                    print('     %.5f' % value_list[idx], end=' ')
                elif isinstance(value_list[idx], int):
                    print('     %7d' % value_list[idx], end=' ')
                else:
                    print('%12s' % str(value_list[idx]), end=' ')

                if idx + 1 == len(value_list):
                    end = True
                    break
            print('')
        row += 1
        print('')


# def print_summary(Dict, num_classes):
#     """Print metric results of each class.
#     Args:
#         Dict(dict(metric name=metric value))
#         num_classes
#     """
#
#     header = []
#     for key, data in Dict.items():
#         header.append(key)
#     table_data = [header]
#     for j in range(num_classes):
#         row_data = []
#         for data in Dict.values():
#             if isinstance(data[j], float):
#                 str_data = '{:.4f}'.format(data[j])
#             elif isinstance(data[j], int):
#                 str_data = '{:6d}'.format(data[j])
#             else:
#                 str_data = '{:6s}'.format(str(data[j]))
#             row_data.append(str_data)
#         table_data.append(row_data)
#
#     mean = ['Mean']
#     for data in Dict.values():
#         if isinstance(data[0], (int, float)):
#             ave = data.mean()
#         # else:
#         #     ave = 0.0
#             mean.append('{:.4f}'.format(ave))
#     table_data.append(mean)
#
#     table = AsciiTable(table_data)
#     table.inner_footing_row_border = True
#     print(table.table)
#
#
# def eval_class_interaction(results, gt_labels, dataset=None):
#     num_classes = results.shape[1]
#
#     if dataset is None:
#         label_names = [str(i) for i in range(1, num_classes + 1)]
#     elif mmcv.is_str(dataset):
#         label_names = get_classes(dataset)
#     else:
#         label_names = dataset
#
#     results = sigmoid(results)
#     co_score = [[] for _ in range(num_classes)]
#     co_num = [[] for _ in range(num_classes)]
#     for result, gt_label in zip(results, gt_labels):
#         for cla in range(num_classes):
#             if gt_label[cla]:
#                 co_score[cla].append(result[cla] * gt_label)
#                 co_num[cla].append(gt_label)
#
#     for cla in range(num_classes):
#         co_score[cla] = np.sum(co_score[cla],0)
#         co_num[cla] = np.sum(co_num[cla],0)
#         co_score[cla] /= co_num[cla]
#
#     for i in range(num_classes):
#         for j in range(num_classes):
#             if co_num[i][j] > 0:
#                 print('{:.3f}'.format(co_score[i][j]), end=' ')
#             else:
#                 print('{:5s}'.format(' --- '), end = ' ')
#         print('')
#
# def sigmoid(x):
#     if isinstance(x,list):
#         x = np.asarray(x)
#     return 1 / (1 + np.exp(-x))
