import numpy as np
from terminaltables import AsciiTable
import mmcv
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


import os.path as osp

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


def display_summary(Dict, num_classes):
    """Print metric results of each class.
    Args:
        Dict(dict(metric name=metric value))
        num_classes
    """

    header = []
    for key, data in Dict.items():
        header.append(key)
    table_data = [header]
    for j in range(num_classes):
        row_data = []
        for data in Dict.values():
            if isinstance(data[j], (float,np.float32)):
                str_data = '{:.4g}'.format(data[j])
            elif isinstance(data[j], (int,np.int64,np.int32)):
                str_data = '{:6d}'.format(data[j])
            else:
                str_data = '{:6s}'.format(str(data[j]))
            row_data.append(str_data)
        table_data.append(row_data)

    mean = ['Mean']
    for i, data in enumerate(Dict.values()):
        if i == 0:
            continue
        if isinstance(data[0], (int, float,np.float32)):
            ave = np.asarray(data).mean()
        # else:
        #     ave = 0.0
            mean.append('{:.4f}'.format(ave))
        else:
            mean.append(' --- ')
    table_data.append(mean)

    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print(table.table)

def sigmoid(x):
    if isinstance(x,list):
        x = np.asarray(x)
    return 1 / (1 + np.exp(-x))

def plotCM(matrix, savename, classes=None ):
    """classes: a list of class names"""
    # Normalize by row
    if classes is None:
        classes = [i for i in range(np.shape(matrix)[0])]
    matrix = matrix.astype(np.float)
    matmax = matrix.max()
    matrix = matrix/matmax

    matrix = 0.5 * matrix + 0.5*(matrix>0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix,cmap=plt.get_cmap('plasma'))
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    # for i in range(matrix.shape[0]):
    #     ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
    ax.set_xticklabels([''] + classes, rotation=90)
    ax.set_yticklabels([''] + classes)
    # save
    # plt.show()
    plt.savefig(savename)


def plotXY(x, y, savename, log=False):
    if not isinstance(y[0],(list,np.ndarray)):
       y = [y]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, y_ in enumerate(y):
        x,y_ = lists_to_arrays([x,y_])
        if log:
            y_ = np.log(y_)
        ax.scatter(x,y_,s=1)#,color = factor_cmap('fruits', palette=Spectral6, factors=y))
    plt.savefig(savename)

def plotBAR(co_num, savename):
    co_num = np.asarray(co_num)
    num_classes = co_num.shape[0]
    names = []
    pairs = []
    for i in range(num_classes):
        for j in range(i):
            names.append('%d_%d'%(i,j))
            pairs.append(co_num[i][j])
    names = np.asarray(names)
    pairs = np.asarray(pairs)
    rank = np.argsort(-pairs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(names[rank],pairs[rank])
    plt.savefig(savename)

def plotHIST(datalist, namelist=None, savename='hist.jpg'):
    if not isinstance(datalist,list):
        datalist = [datalist]
    if namelist is None:
        namelist = np.arange(len(datalist))
    elif not isinstance(namelist,list):
        namelist = [namelist]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, data in enumerate(datalist):
        ax.hist(data, bins=10, label=str(namelist[i]), alpha=0.5)
        ax.legend(loc=1)
        # if i == 0:
        #     ax.hist(data, bins=30, label=str(namelist[i]))
        # else:
        #     ax = ax.twinx()
        #     ax.hist(data, bins=30)
    i = 0
    while osp.exists(savename):
        i += 1
        savename = savename[:-4] + '{}.jpg'.format(i)
    plt.savefig(savename)

def plotBAR_2(co_num, self_acc, both_acc, savename, baseline_path=None):
    co_num, self_acc, both_acc = lists_to_arrays([co_num, self_acc, both_acc])
    num_classes = np.shape(co_num)[0]
    co_num = co_num.astype(np.int)
    num = []
    y_min = []
    y_max = []
    y_mean = []
    y_both = []
    max_freq = int(np.max(co_num))
    freq = np.arange(0, max_freq + 1)
    accs = [[] for _ in range(max_freq + 1)]
    for i in range(num_classes):
        for j in range(i - 1):
            if co_num[i][j] > 0:
                num.append(co_num[i][j])
                accs[co_num[i][j]].append([min(self_acc[i][j], self_acc[j][i]),
                                           max(self_acc[i][j], self_acc[j][i]),
                                           np.mean([self_acc[i][j], self_acc[j][i]]),
                                           both_acc[i][j]
                                           ])
    for f in range(max_freq + 1):
        if accs[f]:
            accs[f] = np.mean(accs[f],0)
    for n in num:
        y_min.append(accs[n][0])
        y_max.append(accs[n][1])
        y_mean.append(accs[n][2])
        y_both.append(accs[n][3])

    num, y_min, y_max, y_mean, y_both = lists_to_arrays([num, y_min, y_max, y_mean, y_both])
    rank = np.argsort(-num)
    x = np.arange(len(num))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax1.plot(x, y_min[rank], '-',linewidth=1,color='#43CD80', label='y_min')
    # ax1.plot(x, y_max[rank], '-', linewidth=1,color='#EEEE00',label='y_max')
    # ax1.plot(x, y_mean[rank], '-', linewidth=1,color='#CD9B9B',label='y_mean')
    ax1.plot(x, y_both[rank], '-', linewidth=1,color='#CD5555',label='y_both')
    ax1.legend(loc=1)
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('accuracy')

    ax2 = ax1.twinx()
    plt.bar(x, num[rank], width=0.85, color='skyblue')
    ax2.set_ylabel('frequency')

    if baseline_path is not None:
        baseline_acc = mmcv.load(baseline_path)
        baseline_both_acc=baseline_acc['both_acc']
    plt.savefig(savename)



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

