from __future__ import division

import re
from collections import OrderedDict

import torch
from mmcv.runner import EpochBasedRunner, DistSamplerSeedHook, obj_from_dict
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner.checkpoint import load_checkpoint

from mllt import datasets
from mllt.core import DistOptimizerHook, DistEvalmAPHook
from mllt.datasets import build_dataloader
from .env import get_root_logger
from mmcv.runner import Hook


class SamplerSeedHook(Hook):

    def before_epoch(self, runner):
        runner.data_loader.sampler.set_epoch(runner.epoch)


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_classifier(model,
                     dataset,
                     cfg,
                     distributed=False,
                     validate=False,
                     logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        raise NotImplementedError
        _dist_train(model, dataset, cfg, validate=validate, logger=logger)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate, logger=logger)


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(
            optimizer_cfg, torch.optim, dict(params=model.parameters()))
        # ''' modify lr in different parts of the network ~
        # '''
        params = []
        # for name, param in model.named_parameters():
        #     param_group = {'params': [param]}
        #     if 'neck' in name:
        #         param_group['lr_mult'] = 0.0001
        #     else:
        #         param_group['lr_mult'] = 1
        #     params.append(param_group)
        # print('>>> Original !')
        # for k, v in dict(params=model.parameters()).items():
        #     print(k, v)
        # print('\033[1;35m>>> Learning rate Modified !\033[0;0m')
        # # for p in params:
        # #     print(p['params'],p['lr_mult'])
        # optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        # return optimizer_cls(params, **optimizer_cfg)
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
                or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        param_group_cfg = paramwise_options.get('param_group_cfg', dict())
        freeze_params = paramwise_options.get('freeze_params', [])
        params = []
        for name, param in model.named_parameters():
            part = name.split('.')[0]
            if part in freeze_params:
                param.requires_grad = False
            if not param.requires_grad:
                continue

            lr_mult = param_group_cfg.get(part, 1.)
            param_group = {'params': [param]}
            param_group['lr_mult'] = lr_mult
            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def _dist_train(model, dataset, cfg, validate=False, logger=None):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True)
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = EpochBasedRunner(model, batch_processor, optimizer, cfg.work_dir, logger)
    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(DistEvalmAPHook(val_dataset_cfg, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        load_modules = cfg.get('load_modules', [])
        load_certain_checkpoint(runner.model, runner.logger, cfg.load_from, load_modules)

        # runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False, logger=None):
    # prepare data loaders
    sampler_cfg = cfg.data.get('sampler_cfg', None)
    sampler = cfg.data.get('sampler', 'Group')
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False,
            sampler=sampler,
            sampler_cfg=sampler_cfg)
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = EpochBasedRunner(model, batch_processor, optimizer, cfg.work_dir, logger)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(SamplerSeedHook())

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        load_modules = cfg.get('load_modules', [])
        load_certain_checkpoint(runner.model, runner.logger, cfg.load_from, load_modules)
        #runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

def load_certain_checkpoint(model, logger, filename, load_modules = ['backbone'], map_location='cpu', strict=False, display=False):
    logger.info('>>> load checkpoint from %s', filename)
    checkpoint = torch.load(filename, map_location=map_location)
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    if hasattr(model, 'module'):
        module = model.module
    else:
        module = model
    unexpected_keys = []
    ignored_keys = []
    own_state = module.state_dict()
    # i = 0
    # for name in own_state.keys():
    #     print(name)
    #     i+=1
    # print(i)
    # i = 0
    # for name in state_dict.keys():
    #     print(name)
    #     i += 1
    # print(i)
    # exit()
    for name, param in state_dict.items():

        if load_modules:
            flag = 0
            for load_module in load_modules:
                if load_module in name:
                    flag = 1
        else:
            flag=1
        if not flag:
            ignored_keys.append(name)
            continue
        if name not in own_state:
            unexpected_keys.append(name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        try:
            own_state[name].copy_(param)
        except Exception:
            raise RuntimeError(
                'While copying the parameter named {}, '
                'whose dimensions in the model are {} and '
                'whose dimensions in the checkpoint are {}.'.format(
                    name, own_state[name].size(), param.size()))
    missing_keys = set(own_state.keys()) - set(state_dict.keys())

    err_msg = []
    if unexpected_keys:
        err_msg.append('>>> unexpected key in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('>>> missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))
    if ignored_keys:
        err_msg.append('>>> mismatched key in source state_dict: {}\n'.format(
            ', '.join(ignored_keys)))
    err_msg = '\n'.join(err_msg)
    if err_msg:
        if strict:
            raise RuntimeError(err_msg)
        elif display:
            if logger is not None:
                logger.warn(err_msg)
            else:
                print(err_msg)
