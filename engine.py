# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import torch
from einops import rearrange
from tqdm import tqdm

import utils
from losses import HuberLoss
from utils import lab2rgb, psnr, rgb2lab
import numpy as np

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None, exp_name=None):#, flops=1):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    # sparsity
    token_list = []
    channel_list = []
    def _report_sparsity(m):
        classname = m.__class__.__name__
        if isinstance(m, poolformer.PoolFormerBlock):
            token_list.append(torch.cat(m.token_list, 0).mean(dim=(1,2)))
            m.token_list = []#torch.tensor([0])

    # loss_func = nn.MSELoss()
    loss_func = HuberLoss()

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step % 1000 == 0:
            print(exp_name)
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_hinted_pos = batch

        images = images.to(device, non_blocking=True)
        # bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)

        # Lab conversion and normalizatoin
        images = rgb2lab(images, 50, 100, 110)  # l_cent, l_norm, ab_norm
        B, C, H, W = images.shape
        h, w = H // patch_size, W // patch_size

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            images_patch = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
            labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

        with torch.cuda.amp.autocast():
            outputs = model(images, bool_hinted_pos)  # ! images has been changed (in-place ops)
            # if flops!=1:
            #     model.apply(_report_sparsity)
            #     token_mean = torch.cat(token_list, 0).mean()
            #     token_list = []
            #     token_flops_loss = ((token_mean - flops)**2).mean()

            outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

            # Loss is calculated only with the ab channels
            loss = loss_func(input=outputs, target=labels[:, :, :, 1:])

            # if flops!=1:
            #     loss += token_flops_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
             patch_size: int = 16, log_writer=None, val_hint_list=[10]):#, flops=1):

    token_list=[]
    channel_list=[]
    def _report_sparsity(m):
        classname = m.__class__.__name__
        if isinstance(m, poolformer.PoolFormerBlock):
            token_list.append(torch.cat(m.token_list, 0).mean(dim=(1,2)))
            m.token_list = []#torch.tensor([0])

    model.eval()
    header = 'Validation'

    psnr_sum = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    num_validated = 0
    with torch.no_grad():
        for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
            # assign learning rate & weight decay for each step
            images, bool_hints = batch
            B, _, H, W = images.shape
            h, w = H // patch_size, W // patch_size

            images = images.to(device, non_blocking=True)
            # Lab conversion and normalizatoin
            images_lab = rgb2lab(images)
            # calculate the predict label
            images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
            labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

            for idx, count in enumerate(val_hint_list):
                bool_hint = bool_hints[:, idx].to(device, non_blocking=True).flatten(1).to(torch.bool)
                # bool_hint = bool_hints.to(device, non_blocking=True).to(torch.bool)

                with torch.cuda.amp.autocast():
                    outputs = model(images_lab.clone(), bool_hint.clone())
                    outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
                    # if flops!=1:
                    #     model.apply(_report_sparsity)

                pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
                pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                                          h=h, w=w, p1=patch_size, p2=patch_size)
                pred_imgs = lab2rgb(pred_imgs_lab)

                _psnr = psnr(images, pred_imgs) * B
                psnr_sum[count] += _psnr.item()
            num_validated += B

        psnr_avg = dict()
        for count in val_hint_list:
            psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated
        # if flops!=1:
        #     print('token sparsity : ', torch.cat(token_list, 0).mean())
        torch.cuda.synchronize()

        if log_writer is not None:
            log_writer.update(head="psnr", **psnr_avg)
    return psnr_avg


def train_pruned_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None, exp_name=None, flops=1):#, flops=1):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    # sparsity
    layer_mask_list = []

    # loss_func = nn.MSELoss()
    loss_func = HuberLoss()

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step % 1000 == 0:
            print(exp_name)
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_hinted_pos = batch

        images = images.to(device, non_blocking=True)
        # bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)

        # Lab conversion and normalizatoin
        images = rgb2lab(images, 50, 100, 110)  # l_cent, l_norm, ab_norm
        B, C, H, W = images.shape
        h, w = H // patch_size, W // patch_size

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            images_patch = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
            labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

        with torch.cuda.amp.autocast():
            outputs, layer_mask = model(images, bool_hinted_pos)  # ! images has been changed (in-place ops)
            outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

            # Loss is calculated only with the ab channels
            loss = loss_func(input=outputs, target=labels[:, :, :, 1:])

            # block layer flops
            # layer_mask shape : (BS, n_layer, 1)
            loss += (layer_mask.mean()-flops)**2

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate_pruned(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
             patch_size: int = 16, log_writer=None, val_hint_list=[10]):#, flops=1):

    layer_mask_list = []

    model.eval()
    header = 'Validation'

    psnr_sum = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    num_validated = 0
    with torch.no_grad():
        for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
            # assign learning rate & weight decay for each step
            images, bool_hints = batch
            B, _, H, W = images.shape
            h, w = H // patch_size, W // patch_size

            images = images.to(device, non_blocking=True)
            # Lab conversion and normalizatoin
            images_lab = rgb2lab(images)
            # calculate the predict label
            images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
            labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

            for idx, count in enumerate(val_hint_list):
                bool_hint = bool_hints[:, idx].to(device, non_blocking=True).flatten(1).to(torch.bool)
                # bool_hint = bool_hints.to(device, non_blocking=True).to(torch.bool)

                with torch.cuda.amp.autocast():
                    outputs, layer_mask = model(images_lab.clone(), bool_hint.clone())
                    outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
                    layer_mask_list.append(layer_mask)


                pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
                pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                                          h=h, w=w, p1=patch_size, p2=patch_size)
                pred_imgs = lab2rgb(pred_imgs_lab)

                _psnr = psnr(images, pred_imgs) * B
                psnr_sum[count] += _psnr.item()
            num_validated += B

        psnr_avg = dict()
        for count in val_hint_list:
            psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated

        print('layer sparsity : ', torch.cat(layer_mask_list, 0).mean().cpu().detach().numpy())
        torch.cuda.synchronize()

        if log_writer is not None:
            log_writer.update(head="psnr", **psnr_avg)
    return psnr_avg




def train_exit_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None, exp_name=None):#, flops=1):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    # sparsity
    token_list = []
    channel_list = []
    def _report_sparsity(m):
        classname = m.__class__.__name__
        if isinstance(m, poolformer.PoolFormerBlock):
            token_list.append(torch.cat(m.token_list, 0).mean(dim=(1,2)))
            m.token_list = []#torch.tensor([0])

    # loss_func = nn.MSELoss()
    loss_func = HuberLoss()

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step % 1000 == 0:
            print(exp_name)
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_hinted_pos = batch

        images = images.to(device, non_blocking=True)
        # bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)

        # Lab conversion and normalizatoin
        images = rgb2lab(images, 50, 100, 110)  # l_cent, l_norm, ab_norm
        B, C, H, W = images.shape
        h, w = H // patch_size, W // patch_size

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            images_patch = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
            labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

        with torch.cuda.amp.autocast():
            aux_outputs_list = model(images, bool_hinted_pos)  # ! images has been changed (in-place ops)

            # if flops!=1:
            #     model.apply(_report_sparsity)
            #     token_mean = torch.cat(token_list, 0).mean()
            #     token_list = []
            #     token_flops_loss = ((token_mean - flops)**2).mean()
            outputs = aux_outputs_list[-1]
            outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
            # Loss is calculated only with the ab channels
            loss = loss_func(input=outputs, target=labels[:, :, :, 1:])

            # aux outputs
            for aux_output in aux_outputs_list[:-1]:
                aux_output = rearrange(aux_output, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

                loss += loss_func(input=aux_output, target=labels[:, :, :, 1:])

            # if flops!=1:
            #     loss += token_flops_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate_exit(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
             patch_size: int = 16, log_writer=None, val_hint_list=[10]):#, flops=1):

    token_list=[]
    channel_list=[]
    def _report_sparsity(m):
        classname = m.__class__.__name__
        if isinstance(m, poolformer.PoolFormerBlock):
            token_list.append(torch.cat(m.token_list, 0).mean(dim=(1,2)))
            m.token_list = []#torch.tensor([0])

    model.eval()
    header = 'Validation'

    psnr_sum = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    psnr_sum_aux1 = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    psnr_sum_aux2 = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    psnr_sum_aux3 = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    psnr_sum_aux4 = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    psnr_sum_aux5 = dict(zip(val_hint_list, [0.] * len(val_hint_list)))

    num_validated = 0
    with torch.no_grad():
        for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
            # assign learning rate & weight decay for each step
            images, bool_hints = batch
            B, _, H, W = images.shape
            h, w = H // patch_size, W // patch_size

            images = images.to(device, non_blocking=True)
            # Lab conversion and normalizatoin
            images_lab = rgb2lab(images)
            # calculate the predict label
            images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
            labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

            for idx, count in enumerate(val_hint_list):
                bool_hint = bool_hints[:, idx].to(device, non_blocking=True).flatten(1).to(torch.bool)
                # bool_hint = bool_hints.to(device, non_blocking=True).to(torch.bool)

                with torch.cuda.amp.autocast():
                    aux_output1, aux_output2, aux_output3, aux_output4, aux_output5, outputs = model(images_lab.clone(), bool_hint.clone())
                    outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
                    aux_output1 = rearrange(aux_output1, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
                    aux_output2 = rearrange(aux_output2, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
                    aux_output3 = rearrange(aux_output3, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
                    aux_output4 = rearrange(aux_output4, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
                    aux_output5 = rearrange(aux_output5, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

                    # if flops!=1:
                    #     model.apply(_report_sparsity)

                # pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
                # pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                #                           h=h, w=w, p1=patch_size, p2=patch_size)
                # pred_imgs = lab2rgb(pred_imgs_lab)

                # main
                pred_imgs = output2rgb(outputs, labels, patch_size, h, w)
                # aux
                pred_imgs_aux1 = output2rgb(aux_output1, labels, patch_size, h, w)
                pred_imgs_aux2 = output2rgb(aux_output2, labels, patch_size, h, w)
                pred_imgs_aux3 = output2rgb(aux_output3, labels, patch_size, h, w)
                pred_imgs_aux4 = output2rgb(aux_output4, labels, patch_size, h, w)
                pred_imgs_aux5 = output2rgb(aux_output5, labels, patch_size, h, w)

                # evaluate the psnr metric
                _psnr = psnr(images, pred_imgs) * B
                psnr_sum[count] += _psnr.item()
                ## aux
                _psnr = psnr(images, pred_imgs_aux1) * B
                psnr_sum_aux1[count] += _psnr.item()
                _psnr = psnr(images, pred_imgs_aux2) * B
                psnr_sum_aux2[count] += _psnr.item()
                _psnr = psnr(images, pred_imgs_aux3) * B
                psnr_sum_aux3[count] += _psnr.item()
                _psnr = psnr(images, pred_imgs_aux4) * B
                psnr_sum_aux4[count] += _psnr.item()
                _psnr = psnr(images, pred_imgs_aux5) * B
                psnr_sum_aux5[count] += _psnr.item()

            num_validated += B

        psnr_avg = dict()
        """psnr 계산할방법 생각하기"""
        for count in val_hint_list:
            psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated
            psnr_avg[f'psnr_aux1@{count}'] = psnr_sum_aux1[count] / num_validated
            psnr_avg[f'psnr_aux2@{count}'] = psnr_sum_aux2[count] / num_validated
            psnr_avg[f'psnr_aux3@{count}'] = psnr_sum_aux3[count] / num_validated
            psnr_avg[f'psnr_aux4@{count}'] = psnr_sum_aux4[count] / num_validated
            psnr_avg[f'psnr_aux5@{count}'] = psnr_sum_aux5[count] / num_validated

        # if flops!=1:
        #     print('token sparsity : ', torch.cat(token_list, 0).mean())
        torch.cuda.synchronize()

        if log_writer is not None:
            log_writer.update(head="psnr", **psnr_avg)
            
    return psnr_avg


def train_decision_exit_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None, exp_name=None, d_model=None, flops=1):#, flops=1):
    d_model.train()
    model.eval()
    # model.train()


    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    # loss_func = nn.MSELoss()
    loss_func = HuberLoss()
    # flops_list = torch.tensor([0.42, 0.62, 0.82, 1.02, 1.23, 1.43]).view(1,1,1,6).to(device).detach()
    flops_list = torch.tensor([1.17, 1.93, 2.69, 3.44, 4.19, 4.94]).view(1,1,1,6).to(device).detach()
    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step % 1000 == 0:
            print(exp_name)
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_hinted_pos = batch

        images = images.to(device, non_blocking=True)
        # bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)

        # Lab conversion and normalizatoin
        images = rgb2lab(images, 50, 100, 110)  # l_cent, l_norm, ab_norm
        B, C, H, W = images.shape
        h, w = H // patch_size, W // patch_size

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            images_patch = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
            labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

        with torch.cuda.amp.autocast():
            aux_outputs_list = model(images, bool_hinted_pos)  # ! images has been changed (in-place ops)
            aux_outputs_list = torch.stack(aux_outputs_list, dim=-1) # (BS, N, C, len_outputs)
            # decision
            mask = d_model(images) # (BS, 1, 1, 6)

            outputs = (aux_outputs_list*mask).sum(-1)

            outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
            # Loss is calculated only with the ab channels
            loss = loss_func(input=outputs, target=labels[:, :, :, 1:])

            # flops loss
            loss+= (((flops_list * mask).sum(-1).mean() - flops) **2).mean()



        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate_decision_exit(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
             patch_size: int = 16, log_writer=None, val_hint_list=[10], d_model=None):

    model.eval()
    d_model.eval()
    header = 'Validation'

    psnr_sum = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    psnr_sum_aux1 = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    psnr_sum_aux2 = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    psnr_sum_aux3 = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    psnr_sum_aux4 = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    psnr_sum_aux5 = dict(zip(val_hint_list, [0.] * len(val_hint_list)))

    num_validated = 0
    # flops_list = np.array([0.42, 0.62, 0.82, 1.02, 1.23, 1.43])
    flops_list = np.array([1.17, 1.93, 2.69, 3.44, 4.19, 4.94])

    number_list = np.array([0., 0., 0., 0., 0., 0.])
    with torch.no_grad():
        for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
            # assign learning rate & weight decay for each step
            images, bool_hints = batch
            B, _, H, W = images.shape
            h, w = H // patch_size, W // patch_size

            images = images.to(device, non_blocking=True)
            # Lab conversion and normalizatoin
            images_lab = rgb2lab(images)
            # calculate the predict label
            images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
            labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

            for idx, count in enumerate(val_hint_list):
                bool_hint = bool_hints[:, idx].to(device, non_blocking=True).flatten(1).to(torch.bool)
                # bool_hint = bool_hints.to(device, non_blocking=True).to(torch.bool)

                with torch.cuda.amp.autocast():
                    outputs_list = model(images_lab.clone(), bool_hint.clone())
                    outputs_list = torch.stack(outputs_list, dim=-1) #(BS, N, C, 6)
                    mask = d_model(images_lab.clone()) # (bs, 1, 1, 6)
                    outputs = (outputs_list*mask).sum(-1)
                    
                    outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

                    number_list += mask.sum(dim=(0,1,2)).cpu().detach().numpy()


                # main
                pred_imgs = output2rgb(outputs, labels, patch_size, h, w)

                # evaluate the psnr metric
                _psnr = psnr(images, pred_imgs) * B
                psnr_sum[count] += _psnr.item()

            num_validated += B

        # related flops 
        print('number of sample at each aux clf', number_list)
        print('All flops:', (flops_list * number_list).sum()/number_list.sum())

        psnr_avg = dict()
        """psnr 계산할방법 생각하기"""
        for count in val_hint_list:
            psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated

        # if flops!=1:
        #     print('token sparsity : ', torch.cat(token_list, 0).mean())
        torch.cuda.synchronize()

        if log_writer is not None:
            log_writer.update(head="psnr", **psnr_avg)
            
    return psnr_avg




def output2rgb(outputs, labels, patch_size, h, w):
    pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
    pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                                h=h, w=w, p1=patch_size, p2=patch_size)
    pred_imgs = lab2rgb(pred_imgs_lab)  
    return pred_imgs