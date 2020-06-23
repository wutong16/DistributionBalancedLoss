import argparse
import os
import shutil
import tempfile
import torch
import resource
import torch.distributed as dist
import mmcv
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import sys
sys.path.append(os.getcwd())

from mllt.apis import init_dist
from mllt.datasets import build_dataloader
from mllt.models import build_classifier
from mllt.core.evaluation.eval_tools import * #eval_paps_from_file, print_summary,eval_class_interaction
from mllt.core.evaluation.mean_ap import eval_map
from sklearn.metrics.pairwise import pairwise_distances

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


def single_gpu_get_feat(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset),bar_width=20)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            _ , feat = model(return_loss=False, rescale=not show, **data)
        results.append(feat.cpu().numpy())
        if show:
            model.module.show_result(data, feat, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_get_feat(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset),bar_width=20)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            _ , feat = model(return_loss=False, rescale=True, **data)
        results.append(feat.cpu().numpy())
        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results

def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results

        ordered_results = []
        for res in zip(*part_list):
           ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

# make a dataset list with both test and training set
# the training set has to maintain its get_ann_info() properties
def make_dataset_list(cfg):
    cfg.data.test.test_mode = True
    test_dataset = build_dataset(cfg.data.test)

    if cfg.data.train.get('dataset',None) is not None:
        train_cfg = cfg.data.train.dataset
    else:
        train_cfg = cfg.data.train
    train_cfg.test_mode = True
    train_cfg.extra_aug = None
    train_cfg.flip_ratio = 0
    train_dataset = build_dataset(train_cfg)

    if isinstance(train_dataset, ConcatDataset):
        train_datasets = train_dataset.datasets
    else:
        train_datasets = [train_dataset]

    dataset_list = [test_dataset]
    for train_dataset in train_datasets:
        dataset_list.append(train_dataset)

    return dataset_list
def parse_args():
    parser = argparse.ArgumentParser(description='extract features')
    parser.add_argument(
        'config', help='test config file path')
    parser.add_argument(
        'checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out', help='output result file')
    parser.add_argument(
        '--eval', type=str, nargs='+', choices=['mAP', 'multiple'],
        default=['multiple'], help='eval metrics')
    parser.add_argument(
        '--show', default=True, help='show results')
    parser.add_argument(
        '--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    parser.add_argument(
        '--local_rank', type=int, default=0)
    parser.add_argument(
        '--testset_only', type=bool, default=True, help='only eval test set')
    parser.add_argument(
        '--from_file', action='store_true', help='load network output results')
    parser.add_argument('--job', type=str, default='eval_feat')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def save_feat(args=None):
    if args is None:
        args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.work_dir = osp.abspath(osp.join(args.checkpoint, '..'))

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    epoch = args.checkpoint.split('.')[-2].split('_')[-1]
    if args.out is None:
        args.out = osp.join(cfg.work_dir, 'gt_and_feats_e{}.pkl'.format(epoch))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    rank, _ = get_dist_info()
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # build the dataloader
    if not cfg.model.get('savefeat', False):
        setattr(cfg.model, "savefeat", True)
    model = build_classifier(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    dataset_list = make_dataset_list(cfg)
    if osp.exists(args.out) and args.from_file:
        savedata = mmcv.load(args.out)
    else:
        savedata = [dict() for _ in range(len(dataset_list))]

        for d, dataset in enumerate(dataset_list):
            data_loader = build_dataloader(
                dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)

            gt_labels = []
            for i in range(len(dataset)):
                ann = dataset.get_ann_info(i)
                gt_labels.append(ann['labels'])

            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = dataset.CLASSES

            if not distributed:
                model = MMDataParallel(model, device_ids=[0])
                outputs = single_gpu_get_feat(model, data_loader, args.show)
            else:
                model = MMDistributedDataParallel(model.cuda())
                outputs = multi_gpu_get_feat(model, data_loader, args.tmpdir)

            if rank == 0:
                features = np.vstack(outputs)
                savedata[d].update(features=features, gt_labels=gt_labels)

        if rank == 0:
            print('\nsaving feats to {}'.format(args.out))
            mmcv.dump(savedata, args.out)
    # change back to normal model
    model = build_classifier(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.job == 'pnorm':
        save_outputs = [dict() for _ in range(len(dataset_list))]
        for d in range(len(dataset_list)):
            features, gt_labels = savedata[d]['features'], savedata[d]['gt_labels']
            outputs = test_pnorm(features, head=model.head, p=2)
            save_outputs[d] = dict(outputs=outputs, gt_labels=gt_labels)
        mmcv.dump(save_outputs, osp.join(cfg.work_dir, 'gt_and_results_e{}.pkl'.format(epoch)))
        print('Test results of pnorm saved at {}'.format(osp.join(cfg.work_dir, 'gt_and_results_e{}.pkl'.format(epoch))))
    elif args.job == 'eval_feat':
        if osp.exists(osp.join(cfg.work_dir, 'gt_and_results_e{}.pkl'.format(epoch))):
            test_results = mmcv.load(osp.join(cfg.work_dir, 'gt_and_results_e{}.pkl'.format(epoch)))[0]['outputs']
            train_results = mmcv.load(osp.join(cfg.work_dir, 'gt_and_results_e{}.pkl'.format(epoch)))[1]['outputs']
        else:
            test_results = None
            train_results = None
        # test_proto(train_features=savedata[1]['features'], test_features=savedata[0]['features'],
        #            train_gt_labels=savedata[1]['gt_labels'], test_gt_labels=savedata[0]['gt_labels'],
        #            cla_results=cla_results, save_name=cfg.work_dir + '/proto_compare.jpg')
        # exit()
        eval_feat(train_features=savedata[1]['features'], train_gt_labels=savedata[1]['gt_labels'], train_results=train_results,
                  results=test_results, gt_labels=savedata[0]['gt_labels'], features=savedata[0]['features'],
                  head=model.head, save_name=cfg.work_dir + '/eval_feat.jpg')
    elif args.job == 'eval_centroid':
        assert not args.testset_only
        save_outputs = [dict() for _ in range(len(dataset_list))]
        train_features, train_gt_labels = savedata[1]['features'], savedata[1]['gt_labels']
        for d in range(len(dataset_list)):
            features, gt_labels = savedata[d]['features'], savedata[d]['gt_labels']
            outputs = test_centroids(features, train_features, train_gt_labels)
            save_outputs[d] = dict(outputs=outputs, gt_labels=gt_labels)
        mmcv.dump(save_outputs, osp.join(cfg.work_dir, 'gt_and_results_e{}.pkl'.format(epoch)))
        print(
            'Test results of centroids saved at {}'.format(osp.join(cfg.work_dir, 'gt_and_results_e{}.pkl'.format(epoch))))
        exit()
    else:
        raise NameError


def eval_train_test(train_gt, test_gt,save_name='train_test_num.jpg'):
    train_gt, test_gt = lists_to_arrays([train_gt, test_gt])
    num_classes = train_gt.shape[1]
    sample_num = np.sum(train_gt, axis=0)
    test_sample_num = np.sum(test_gt, axis=0)
    # remove person
    rank_idx = np.argsort(-sample_num)

    fig, ax = plt.subplots(1, 1)
    ax.bar(range(num_classes), sample_num[rank_idx], alpha=0.5)
    ax1 = ax.twinx()
    ax1.bar(range(num_classes), test_sample_num[rank_idx], color='red', alpha=0.5)
    plt.savefig(save_name)
    print('eval for train and test saved at {}'.format(save_name))


def eval_feat(train_features, train_gt_labels, train_results=None,
              results=None, gt_labels=None, features=None,
              head=None, save_name='./eval_feat.jpg'):
    train_gt_labels, train_features = lists_to_arrays([train_gt_labels, train_features])
    num_classes = train_gt_labels.shape[-1]
    index_per_class = []
    for cla in range(num_classes):
        indexs = np.where(train_gt_labels[:, cla] > 0)[0].tolist()
        index_per_class.append(indexs)
        # print(len(indexs))
    sample_num = np.asarray([ len(indexs) for indexs in index_per_class ])
    rank_idx = np.argsort(-sample_num)
    fig, ax = plt.subplots(1, 1)
    ax.bar(range(num_classes), sample_num[rank_idx], alpha=0.5)
    ax1 = ax.twinx()
    norm_per_cla = np.zeros(num_classes)
    var_per_cla = []
    for cla, indexs in enumerate(index_per_class):
        feature_cla = train_features[indexs]
        corr = np.matmul(feature_cla , feature_cla.transpose())
        diag = corr.diagonal()
        norm_per_cla[cla] = np.mean(np.sqrt(diag))

    # ax1.bar(range(num_classes), norm_per_cla[rank_idx], alpha=0.5)

    if head is not None:
        weight_norm = torch.norm(head.fc_cls.weight.detach(),2,1).numpy()
        bias = head.fc_cls.bias.detach().numpy()
        weight = head.fc_cls.weight.detach().numpy()
        ax1.bar(range(num_classes), weight_norm[rank_idx], alpha=0.5)
        ax1.bar(range(num_classes), bias[rank_idx], alpha=0.5)

        fake_results = np.matmul(train_features, weight.transpose()) + bias
        # print(cls_results[0])
        # print(fake_results[0])
        # exit()
    plt.savefig(save_name)
    print('eval feature results saved at {}'.format(save_name))
    draw_tsne(train_results, train_gt_labels, train_features, results=None, gt_labels=None, features=None)


def plot_embedding(data, label, title='./tsne_example.jpg'):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    num_classes = label.shape[1]
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        for cla in range(num_classes):
            if label[i][cla] == 0:
                continue
            plt.text(data[i, 0], data[i, 1], str(cla), color=plt.cm.Set1(cla / num_classes * 1.0), fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(title)
    print('tsne results saved at {}'.format(title))
    return fig

def draw_tsne(train_results, train_gt_labels, train_features,
              results=None, gt_labels=None, features=None,
              title='./tsne_example.jpg'):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time.time()
    train_data = tsne.fit_transform(train_features)
    sample_num = np.sum(train_gt_labels, 0)
    rank_idx = np.argsort(-sample_num)
    cls_set = np.hstack((rank_idx[:5] , rank_idx[-5:])).tolist()
    names = ['./save_imgs/KDE_tsne_cls{}_num{}.jpg'.format(cla, sample_num[cla]) for cla in cls_set]
    weights = sigmoid(train_results) - 0.5
    logit_distribution_tsne_2d(train_data, weights, cls_set, names)
    # fig = plot_embedding(result, train_gt_labels,title)


def logit_distribution_tsne_2d(data, weights, cls_set=[0,10], names=None):
    import scipy.stats as st
    ax_min, ax_max = np.min(data, 0), np.max(data, 0)
    data = (data - ax_min) / (ax_max - ax_min)*(-2) + 1
    # x_min, x_max = ax_min[0], ax_max[0]
    # y_min, y_max = ax_min[1], ax_max[1]
    x_min, x_max = -1.5 , 1.5
    y_min, y_max = -1.5 , 1.5
    num_classes = weights.shape[1]
    # data = np.random.multivariate_normal((0, 0), [[0.8, 0.05], [0.05, 0.7]], 100)
    x = data[:, 0]
    y = data[:, 1]

    # Peform the kernel density estimate
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    for i, cla in enumerate(cls_set):
        weight = weights[:,cla]
        name = names[i] if names is not None else './2d_tsne_{}.jpg'.format(cla)
        kernel = st.gaussian_kde(values, weights=weight)
        f = np.reshape(kernel(positions).T, xx.shape)

        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # Contourf plot
        cfset = ax.contourf(xx, yy, f, cmap='Blues')
        ## Or kernel density estimate plot instead of the contourf plot
        # ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
        # Contour plot
        cset = ax.contour(xx, yy, f, colors='k')
        # Label plot
        ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel('Y1')
        ax.set_ylabel('Y0')

        plt.savefig(name)


def test_proto(train_features, train_gt_labels, test_features, test_gt_labels, cla_results=None,save_name='proto_compare.jpg'):
    train_features, train_gt_labels, test_features, test_gt_labels = lists_to_arrays(
        [train_features, train_gt_labels, test_features, test_gt_labels])
    num_classes = train_gt_labels.shape[-1]
    len_features = train_features.shape[-1]
    sample_num = np.sum(train_gt_labels, axis=0)
    rank_idx = np.argsort(-sample_num)
    index_per_class = []
    for cla in range(num_classes):
        indexs = np.where(train_gt_labels[:, cla] > 0)[0].tolist()
        index_per_class.append(indexs)
    proto_per_class = np.zeros([num_classes, len_features])
    for cla, indexs in enumerate(index_per_class):
        feature_cla = train_features[indexs]
        proto_per_class[cla] = np.mean(feature_cla, axis=0)

    scores = np.matmul(test_features, proto_per_class.transpose())
    mAP_protp, APs_proto = eval_map(scores, test_gt_labels, None, print_summary=False)
    if cla_results is not None:
        mAP, APs = eval_map(cla_results, test_gt_labels, None, print_summary=False)
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(num_classes), sample_num[rank_idx], alpha=0.5)
    ax1 = ax.twinx()
    ax1.bar(range(num_classes), APs[rank_idx], color='green', alpha=0.4, label='cla')
    ax1.bar(range(num_classes), APs_proto[rank_idx], color='red', alpha=0.4, label='proto')
    plt.savefig(save_name)
    print('eval for train and test saved at {}'.format(save_name))

    eval_long_tail(None, train_gt_labels, scores, test_gt_labels, save_name.split('/')[0], 'eval_lt_proto.jpg')

def test_pnorm(test_features, head=None, p=1):
    test_features = np.asarray(test_features)

    normB = torch.norm(head.fc_cls.weight.detach(),2,1).numpy()
    weights = head.fc_cls.weight.detach().numpy()
    for i in range(weights.shape[0]):
        weights[i] = weights[i] / np.power(normB[i], p)
    cls_scores = np.matmul(test_features, weights.transpose())
    return cls_scores

def test_centroids(test_features, train_features, train_gt_labels, mode="cosine"):
    test_features, train_features, train_gt_labels = lists_to_arrays([test_features, train_features, train_gt_labels])
    num_classes = train_gt_labels.shape[-1]
    protos = [[] for _ in range(num_classes)]
    for gt_label, feature in zip(train_gt_labels, train_features): 
        gt_idx = np.where(gt_label > 0)[0].tolist()
        for i in gt_idx:
            protos[i].append(feature)
    for i in range(num_classes):
        protos[i] = np.mean(protos[i], 0)

    assert mode in ["euclidean", "cosine"]
    dist = pairwise_distances(test_features, protos, metric=mode)
    print(dist.shape)

    return -dist

if __name__ == '__main__':
    save_feat()
