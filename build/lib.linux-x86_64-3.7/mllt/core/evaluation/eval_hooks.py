import torch.distributed as dist
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate
from torch.utils.data import Dataset
from .mean_ap import eval_map
from mllt import datasets
from .eval_tools import *

class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1, split=False):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets, {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.split = split

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()

        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset),bar_width=20)
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result.cpu().numpy()

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(
                runner.work_dir, 'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError

class DistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        gt_labels = []
        for i in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(i)
            labels = ann['labels']
            gt_labels.append(labels)
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        tmp_result_file = osp.join(
            runner.work_dir, 'gt_and_results_e{}.pkl'.format(runner.epoch + 1))
        mmcv.dump(dict(gt_labels=gt_labels,
                       outputs=np.vstack(results),
                       ann_labels=gt_labels),
                  tmp_result_file)
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            np.vstack(results),
            np.array(gt_labels),
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True

        tmp_result_file = osp.join(runner.work_dir, 'temp_mAP.txt')
        mode = 'w' if runner.epoch + 1 == self.interval else 'a'
        with open(tmp_result_file,mode) as f:
            f.write('Ep%-3d mAP%.4f |'%(runner.epoch+1,mean_ap))


