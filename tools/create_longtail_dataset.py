import pickle
import os
import os.path as osp
from pycocotools.coco import COCO
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from scipy.stats import pareto
import numpy as np
import mmcv
import random
from mmcv import mkdir_or_exist
import sys
sys.path.append(os.getcwd())
from mllt.core import get_classes


random.seed(0)

def _savefig(name, dpi=600):
    plt.savefig(name, dpi=dpi)
    print('saving at {}'.format(name))

def draw_pareto_changing_b(b_set, num_classes, max=1000, min=1, head=0.0, tail=0.99, save_name='./pareto_ref.jpg'):
    fig, ax = plt.subplots(1, 1)
    classes = np.linspace(0, num_classes, 10*num_classes)
    for i, b in enumerate(b_set):
        rv = pareto(b)
        classes_x = np.linspace(pareto.ppf(head, b), pareto.ppf(tail, b), 10*num_classes)
        dist = rv.pdf(classes_x) * (max-min) / b + min
        ax.plot(classes, dist, label='alpha={}'.format(b))
        plt.legend()
    ax.set_xlabel('sorted class index')
    ax.set_ylabel('sample numbers')

    _savefig(save_name)

def draw_pareto_changing_maxmin(b, num_classes, max=[1000, 500], min=[50, 10], head=0.0, tail=0.99, save_name='./pareto_ref.jpg'):
    fig, ax = plt.subplots(1, 1)
    classes = np.linspace(0, num_classes, 10*num_classes)
    for i, (max_c, min_c) in enumerate(zip(max, min)):
        rv = pareto(b)
        classes_x = np.linspace(pareto.ppf(head, b), pareto.ppf(tail, b), 10*num_classes)
        dist = rv.pdf(classes_x) / b * (max_c-min_c) + min_c
        ax.plot(classes, dist, label='{}'.format(i))
        plt.legend()
    ax.set_ylim(0, np.max(max)+50)
    ax.set_xlabel('sorted class index')
    ax.set_ylabel('sample numbers')

    _savefig(save_name)

def pareto_dist(b, num_classes, max, min=1, tail=0.99, display=False):
    """ generate a pareto distribution reference dist
    """
    rv = pareto(b)
    classes = range(num_classes)
    classes_x = np.linspace(pareto.ppf(0.0, b), pareto.ppf(tail, b), num_classes)
    dist = rv.pdf(classes_x) * (max-min) / b + min
    dist  = dist.astype(int)
    if display:
        fig, ax = plt.subplots(1, 1)
        ax.bar(classes, dist)
        plt.savefig('./data/longtail/refer_num{:d}_b{:d}_max{:d}-min{:d}.jpg'.format(num_classes, b, max, min))
    return dist

def create_voc_longtail(year=2012,  max=800, b = 6, save_dir ='./appendix/VOCdevkit/', draw=False):
    data = mmcv.load('appendix/VOCdevkit/terse_gt_{}.pkl'.format(year))
    gt_labels = np.asarray(data['gt_labels'])
    img_id2idx = data['img_id2idx']
    idx2img_id = data['idx2img_id']

    save_dir  = save_dir + 'longtail' + str(year) + '/'
    mkdir_or_exist(save_dir)

    num_classes = gt_labels.shape[1]
    sample_num = np.sum(gt_labels, axis=0)
    rank_idx = np.argsort(-sample_num)
    ref_dist = pareto_dist(b, num_classes, max=max, min=1, tail=0.99, display=False)

    ori_idx_dic = {i: [] for i in range(num_classes)}
    tmp_idx_dic = {i: set() for i in range(num_classes)}
    co_labels = [[] for _ in range(num_classes)]
    for i, label in enumerate(gt_labels):
        for idx in np.where(np.asarray(label) == 1)[0]:
            ori_idx_dic[idx].append(i)
            co_labels[idx].append(label)
    co_occurence = np.zeros([num_classes, num_classes])
    co_prob = np.zeros([num_classes, num_classes])
    for i in range(num_classes):
        co_occurence[i] = np.sum(co_labels[i], axis=0)
        co_prob[i] = co_occurence[i] / co_occurence[i][i]
    sample_prob = np.sum(co_prob, axis=0)
    rank_idx = np.argsort(-sample_prob)

    select_idx = set()
    add_new =  np.zeros(num_classes)
    tmp_dist = np.zeros(num_classes)
    pile_dist = []
    for i, idx in enumerate(rank_idx):
        new_sample = int(ref_dist[i] - tmp_dist[idx])
        print('for idx:{:4d}, have {:4d}, want {:4d}, total {:6d}'.format(
            idx, len(tmp_idx_dic[idx]), ref_dist[i], len(set(ori_idx_dic[idx]))))
        if ref_dist[i] > sample_num[idx]:
            print('for class {}, want {}, which is more than the total num of {}'.format(
                idx, ref_dist[i], sample_num[idx]))
        if new_sample < 0:
            this_set = random.sample(tmp_idx_dic[idx], - new_sample)
            for t in this_set:
                add_new[i] -= 1
                select_idx.remove(t)
                tmp_dist -= gt_labels[t]
                gt_idxs = np.where(gt_labels[t] > 0)[0].tolist()
                for gt_idx in gt_idxs:
                    tmp_idx_dic[gt_idx].remove(t)
        elif new_sample > 0:
            this_set = random.sample(set(ori_idx_dic[idx]) - tmp_idx_dic[idx], new_sample)
            for t in this_set:
                add_new[i] += 1
                select_idx.add(t)
                tmp_dist += gt_labels[t]
                gt_idxs = np.where(gt_labels[t] > 0)[0].tolist()
                for gt_idx in gt_idxs:
                    tmp_idx_dic[gt_idx].add(t)
        pile_dist.append(tmp_dist.copy())

    select_labels = []
    select_img_id = []
    class_per_image = np.zeros(num_classes)
    for idx in select_idx:
        select_labels.append(gt_labels[idx])
        select_img_id.append(idx2img_id[idx])
        gt_idxs = np.where(gt_labels[idx] > 0)[0].tolist()
        for gt_idx in gt_idxs:
            class_per_image[gt_idx] += np.sum(gt_labels[idx])
    select_sample_num = np.sum(select_labels, axis=0)
    class_per_image = class_per_image / select_sample_num
    rank_idx = np.argsort(-select_sample_num)

    if draw:
        # train sample number distribution
        fig, ax = plt.subplots(1, 1)
        ax.bar(range(num_classes), sample_num[rank_idx], alpha=0.5)
        ax.bar(range(num_classes), sample_num[rank_idx][1] / max * ref_dist,
               alpha=0.5)
        plt.savefig('./data/longtail/voc_dist_ref.jpg')
        # conditional probability distribution
        fig, ax = plt.subplots(1, 1)
        ax.bar(range(num_classes), sample_prob[rank_idx])
        plt.savefig('./data/longtail/voc_prob.jpg')
        # total samples added each time
        fig, ax = plt.subplots(1, 1)
        ax.bar(range(num_classes), add_new)
        ax.set_xlabel('sorted class index')
        ax.set_ylabel('sample numbers')
        _savefig('./data/longtail/voc_add_samplenum.jpg')
        # new dataset distribution
        fig, ax1 = plt.subplots(1, 1)
        ax1.bar(range(num_classes), select_sample_num[rank_idx])
        for y in [20,100]:
            plt.hlines(y, 0, num_classes, linestyles='dashed', color='brown', linewidth=0.5)
        ax2 = ax1.twinx()
        ax2.plot(range(num_classes), class_per_image[rank_idx], color='brown', alpha=0.8)
        plt.savefig('./data/longtail/voc_select_dist.jpg')
        # per-class samples added each time
        fig, ax = plt.subplots(1, 1)
        for i in range(len(pile_dist)):
            ax.bar(range(num_classes), pile_dist[-i][rank_idx], alpha=0.8)
        ax.set_xlabel('sorted class index')
        ax.set_ylabel('sample numbers')
        plt.savefig('./data/longtail/voc_add_perclass.jpg')

    head_clas, middle_clas, tail_clas = [set(np.where(select_sample_num>=100)[0]),
                                         set(np.where((select_sample_num<100) * (select_sample_num >= 20))[0]),
                                         set(np.where(select_sample_num<20)[0])]
    print('Train set, head classes: {:d}, middle classes: {:d}, tail classes: {:d}'.format(
        len(head_clas), len(middle_clas), len(tail_clas)))

    print('dataset length: {}'.format(len(select_img_id)))

    save_path = save_dir+'img_id.txt'.format(b)
    if osp.exists(save_path):
        print('{} already exists, won\'t overwrite!'.format(save_path))
    else:
        with open(save_path, "w") as f:
            for img_id in select_img_id:
                f.writelines("%s\n" % img_id)
        mmcv.dump(dict(head=head_clas, middle=middle_clas, tail=tail_clas), save_dir+'class_split.pkl')
        print('new dataset saved in {}'.format(save_path))
        print('class split saved in {}'.format(save_dir+'class_split.pkl'))
    return



def create_coco_longtail(year=2017, max=1200, min=1, b = 6, save_dir ='./appendix/coco', draw=False):
    data = mmcv.load('appendix/coco/terse_gt_{}.pkl'.format(year))
    gt_labels = data['gt_labels']
    test_gt_labels = data['test_gt_labels']
    test_samples = np.sum(test_gt_labels, axis=0)
    img_id2idx = data['img_id2idx']
    idx2img_id = data['idx2img_id']

    save_dir  = osp.join(save_dir, 'longtail' + str(year))
    mkdir_or_exist(save_dir)
    category_names = get_classes('coco')
    num_classes = len(category_names)
    sample_num = np.sum(gt_labels, axis=0)
    rank_idx = np.argsort(-sample_num)
    ref_dist = pareto_dist(b, num_classes, max=max, min=min, tail=0.99, display=False)

    ori_idx_dic = { i: [] for i in range(num_classes) }
    tmp_idx_dic = {i: set() for i in range(num_classes)}
    co_labels = [[] for _ in range(num_classes)]
    for i, label in enumerate(gt_labels):
        for idx in np.where(np.asarray(label) == 1)[0]:
            ori_idx_dic[idx].append(i)
            co_labels[idx].append(label)

    co_occurence = np.zeros([num_classes, num_classes])
    co_prob = np.zeros([num_classes, num_classes])
    for i in range(num_classes):
        co_occurence[i] = np.sum(co_labels[i], axis=0)
        co_prob[i] = co_occurence[i] / co_occurence[i][i]

    sample_prob = np.sum(co_prob, axis=0)
    rank_idx = np.argsort(-sample_prob)

    select_idx = set()
    add_new =  np.zeros(num_classes)
    tmp_dist = np.zeros(num_classes)
    pile_dist = []
    for i, idx in enumerate(rank_idx):
        new_sample = int(ref_dist[i] - tmp_dist[idx])
        print('for idx:{:4d}, have {:4d}, want {:4d}, total {:6d}'.format(
            idx, len(tmp_idx_dic[idx]), ref_dist[i], len(set(ori_idx_dic[idx]))))
        if new_sample < 0:
            this_set = random.sample(tmp_idx_dic[idx], - new_sample)
            for t in this_set:
                add_new[i] -= 1
                select_idx.remove(t)
                tmp_dist -= gt_labels[t]
                gt_idxs = np.where(gt_labels[t] > 0)[0].tolist()
                for gt_idx in gt_idxs:
                    tmp_idx_dic[gt_idx].remove(t)
        elif new_sample > 0:
            new_sample = np.minimum(len(set(ori_idx_dic[idx])- tmp_idx_dic[idx]), new_sample)
            this_set = random.sample(set(ori_idx_dic[idx]) - tmp_idx_dic[idx], new_sample)
            for t in this_set:
                add_new[i] += 1
                select_idx.add(t)
                tmp_dist += gt_labels[t]
                gt_idxs = np.where(gt_labels[t] > 0)[0].tolist()
                for gt_idx in gt_idxs:
                    tmp_idx_dic[gt_idx].add(t)
        pile_dist.append(tmp_dist.copy())

    select_labels = []
    select_img_id = []
    class_per_image = np.zeros(num_classes)
    for idx in select_idx:
        select_labels.append(gt_labels[idx])
        select_img_id.append(idx2img_id[idx])
        gt_idxs = np.where(gt_labels[idx] > 0)[0].tolist()
        for gt_idx in gt_idxs:
            class_per_image[gt_idx] += np.sum(gt_labels[idx])
    select_sample_num = np.sum(select_labels, axis=0)
    class_per_image = class_per_image / select_sample_num
    rank_idx = np.argsort(-select_sample_num)
    if draw:
        # train sample number distribution
        fig, ax = plt.subplots(1, 1)
        sample_num[0] = 0
        ax.bar(range(num_classes), sample_num[rank_idx],alpha=0.5)
        ax.bar(range(num_classes), sample_num[rank_idx][1]/max*ref_dist, alpha=0.5)
        _savefig('./data/longtail/coco_dist_ref.jpg')
        # test sample number distribution
        fig, ax = plt.subplots(1, 1)
        test_samples[rank_idx][0] = 0
        ax.bar(range(num_classes), test_samples[rank_idx])
        ax.set_xlabel('sorted class index')
        ax.set_ylabel('test sample numbers')
        _savefig('./coco_test_dist.jpg')
        # total samples added each time
        fig, ax = plt.subplots(1, 1)
        ax.bar(range(num_classes), add_new)
        plt.savefig('./coco_add_samplenum.jpg')
        # probability distribution
        fig, ax = plt.subplots(1, 1)
        ax.bar(range(num_classes), sample_prob[rank_idx])
        plt.savefig('./data/longtail/coco_prob.jpg')
        # new dataset sample number distribution
        fig, ax1 = plt.subplots(1, 1)
        ax1.bar(range(num_classes), select_sample_num[rank_idx])
        for y in [20,100]:
            plt.hlines(y, 0, 80, linestyles='dashed', color='r', linewidth=0.5)
        ax1.plot(range(num_classes), 20 * class_per_image[rank_idx], color='purple', alpha=0.5)
        _savefig('./data/longtail/coco_sel_dist.jpg')
        # per-class samples added each time
        fig, ax = plt.subplots(1, 1)
        for i in range(len(pile_dist)):
            ax.bar(range(num_classes), pile_dist[-i][rank_idx])
        ax.set_xlabel('sorted class index')
        ax.set_ylabel('train sample numbers')
        _savefig('./coco_add_perclass.jpg')

    head_clas, middle_clas, tail_clas = [set(np.where(select_sample_num>=100)[0]),
                                         set(np.where((select_sample_num<100) * (select_sample_num >= 20))[0]),
                                         set(np.where(select_sample_num<20)[0])]
    print('Train set, head classes: {:d}, middle classes: {:d}, tail classes: {:d}'.format(
        len(head_clas), len(middle_clas), len(tail_clas)))
    print('dataset length: {}'.format(len(select_img_id)))

    save_path = osp.join(save_dir, 'img_id.pkl')
    if osp.exists(save_path):
        print('{} already exists, won\'t overwrite!'.format(save_path))
    else:
        with open(save_path, "w") as f:
            for img_id in select_img_id:
                f.writelines("%s\n" % img_id)
        mmcv.dump(dict(head=head_clas, middle=middle_clas, tail=tail_clas), osp.join(save_dir, 'class_split.pkl'))
        print('new dataset saved in {}'.format(save_path))
        print('class split saved in {}'.format(osp.join(save_dir, 'class_split.pkl')))
    return

def lvis_longtail_statistics(file='./appendix/lvis/longtail/statistics.pkl', save_dir='./appendix/lvis/longtail/'):
    data = mmcv.load(file)
    train_gt_labels = np.asarray(data['train_gt_labels'])
    test_gt_labels = np.asarray(data['test_gt_labels'])
    CLASSES = data['CLASSES']
    image_count = data['image_count']

    data = mmcv.load('appendix/lvis/terse_gt.pkl')
    gt_labels = data['gt_labels']
    img_id2idx = data['img_id2idx']
    idx2img_id = data['idx2img_id']

    num_classes = train_gt_labels.shape[1]
    num_images = train_gt_labels.shape[0]

    sample_num = np.sum(train_gt_labels, axis=0)
    test_sample_num = np.sum(test_gt_labels, axis=0)
    rank_idx = np.argsort(-sample_num)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.bar(range(num_classes), sample_num[rank_idx], color='blue', alpha=0.5, label='train')
    for y in [20,100]:
        ax1.hlines(y, 0, len(sample_num), linestyles='dashed', color='r', linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.bar(range(num_classes), test_sample_num[rank_idx], color='orange', alpha=0.5, label='test')
    plt.legend()
    head_clas, middle_clas, tail_clas = [set(np.where(sample_num >= 100)[0]),
                                         set(np.where((sample_num < 100) * (sample_num >= 20))[0]),
                                         set(np.where(sample_num < 20)[0])]
    mmcv.dump(dict(head=head_clas, middle=middle_clas, tail=tail_clas), save_dir + 'class_split.pkl')
    print('class split saved in {}'.format(save_dir + 'train_class_split.pkl'))
    print('Train set, head classes: {:d}, middle classes: {:d}, tail classes: {:d}'.format(
        len(head_clas), len(middle_clas), len(tail_clas)))

    # ---- use the 830 validation classes ----
    val_index = np.where(test_sample_num > 0)[0]
    sel_sample_num = sample_num[val_index]
    sel_test_sample_num = test_sample_num[val_index]
    sel_rank_idx = np.argsort(-sel_sample_num)
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.bar(range(len(sel_sample_num)), sel_sample_num[sel_rank_idx], color='blue', alpha=0.5, label='train')
    ax3.bar(range(len(sel_sample_num)), sel_sample_num[sel_rank_idx], color='blue', alpha=0.5, label='train')
    for y in [20, 100]:
        ax3.hlines(y, 0, len(sel_sample_num), linestyles='dashed', color='r', linewidth=0.5)
    ax4 = ax3.twinx()
    ax4.bar(range(len(sel_sample_num)), sel_test_sample_num[sel_rank_idx], color='orange', alpha=0.5, label='test')
    plt.legend()
    head_clas, middle_clas, tail_clas = [set(np.where(sel_sample_num >= 100)[0]),
                                         set(np.where((sel_sample_num < 100) * (sel_sample_num >= 20))[0]),
                                         set(np.where(sel_sample_num < 20)[0])]
    mmcv.dump(dict(head=head_clas, middle=middle_clas, tail=tail_clas), save_dir + 'class_split.pkl')
    print('class split saved in {}'.format(save_dir + 'class_split.pkl'))
    print('Train set, head classes: {:d}, middle classes: {:d}, tail classes: {:d}'.format(len(head_clas),
                                                                                           len(middle_clas),
                                                                                           len(tail_clas)))
    plt.savefig(osp.join(save_dir, 'lvis_dist.jpg'))
    # assert np.sum(np.asarray(image_count) - sample_num) == 0
    print('train classes:{}'.format(len(np.where(sample_num > 0)[0])))
    print('test classes:{}'.format(len(np.where(test_sample_num > 0)[0])))
    # print(image_count[:20])
    # print(sample_num[:20])

    val_exist = np.zeros(num_classes)
    val_exist[val_index] = 1
    print(val_index[:10])
    print(np.sum(val_exist))

    select_img_id = []
    for idx, label in enumerate(train_gt_labels):
        if np.sum(label * val_exist) > 0:
            select_img_id.append(idx2img_id[idx])
    print('from {} to {} train data'.format(len(idx2img_id), len(select_img_id)))
    if osp.exists(save_dir+'img_id.pkl'):
        print('{} already exists, won\'t overwrite!'.format(save_dir+'img_id.pkl'))
    else:
        with open(save_dir+'img_id.pkl', "w") as f:
            for img_id in select_img_id:
                f.writelines("%d\n" % img_id)
        print('image_id info saved at {}'.format(save_dir+'img_id.pkl'))

    # FIXME: important!! change list to set would change the order!!
    LVIS_CLASSES = CLASSES
    LVIS_SEEN = [CLASSES[i] for i in val_index]
    LVIS_UNZEEN = [CLASSES[i] for i in np.where(test_sample_num == 0)[0]]
    class_data = dict(LVIS_CLASSES=LVIS_CLASSES, LVIS_SEEN=LVIS_SEEN, LVIS_UNSEEN=LVIS_UNZEEN)
    mmcv.dump(class_data, save_dir+'class_data.pkl')
    print('class seen and unseen data saved at {}'.format(save_dir+'class_data.pkl'))

if __name__ == '__main__':
    create_voc_longtail()
    create_coco_longtail()






