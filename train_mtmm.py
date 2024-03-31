import argparse
import datetime
import logging
import os
import time
import warnings
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchshow as ts
import torchvision
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import utils as utils
from data import dataset_EgoGesture, dataset_NvGesture
from models import models_MTMM as TSN_model
from models.spatial_transforms import *
from models.temporal_transforms import *

warnings.filterwarnings("ignore")

os.environ['WANDB_MODE'] = 'disabled'


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='0')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--modal', type=str, default='rgb_depth')
    parser.add_argument('--train_plus_val', action="store_true")
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--amp', action="store_true")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--subset', action="store_true")

    # args for dataloader
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--clip_len', type=int, default=8)

    # args for preprocessing
    parser.add_argument('--initial_scale', type=float, default=1,
                        help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int,
                        help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float,
                        help='Scale step for multiscale cropping')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_steps', type=float, default=[5, 10, 15], nargs="+",
                        help='lr steps for decreasing learning rate')
    parser.add_argument('--clip_gradient', '--gd', type=int,
                        default=20, help='gradient clip')
    parser.add_argument('--shift_div', default=8, type=int)
    parser.add_argument('--is_shift', action="store_true")
    parser.add_argument('--npb', action="store_true")
    parser.add_argument('--pretrain', type=str, default='imagenet')
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--base_model', default='resnet50', type=str)
    parser.add_argument('--dataset', default='EgoGesture', type=str)
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--pretrained', default=None, type=str)
    args = parser.parse_args()
    return args


args = parse_opts()

params = dict()
if args.dataset == 'EgoGesture':
    if args.subset:
        params['num_classes'] = 10
    else:
        params['num_classes'] = 83
elif args.dataset == 'NvGesture':
    params['num_classes'] = 25

params['epoch_num'] = args.epochs
params['batch_size'] = args.batch_size
params['num_workers'] = args.num_workers
params['learning_rate'] = args.lr
params['momentum'] = 0.9
params['weight_decay'] = args.weight_decay
params['display'] = 100
params['log'] = 'log-{}'.format(args.dataset)
params['save_path'] = '{}-{}'.format(args.dataset, args.base_model)
params['clip_len'] = args.clip_len
params['frame_sample_rate'] = 1


annot_path = './data/{}_annotation'.format(args.dataset)
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
device = 'cuda:0'


class EMAWrapper(torch.nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        self.model = deepcopy(model)
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.model.to(self.device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.model.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e,
                     m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def visualize_input(rgb, depth, skeleton):
    print(rgb.min(), rgb.max())
    print(depth.min(), depth.max())
    print(skeleton.min(), skeleton.max())
    input_mean = [.485, .456, .406]
    input_std = [.229, .224, .225]
    rgb_copy = rgb.clone()[0, 0].permute(1, 2, 0).cpu().numpy()
    rgb_copy = rgb_copy * input_std + input_mean
    rgb_copy = np.clip(rgb_copy, 0, 1)
    depth_copy = depth[0, 0].permute(1, 2, 0).squeeze(-1).cpu().numpy()
    skeleton_copy = skeleton[0, 0].permute(1, 2, 0).sum(-1).cpu().numpy()
    plt.imsave(args.model_path + 'rgb.jpg', rgb_copy)
    plt.imsave(args.model_path + 'depth.jpg', depth_copy, cmap='gray')
    plt.imsave(args.model_path + 'skeleton.jpg', skeleton_copy)


def train(model, train_dataloader, epoch, criterion, optimizer, mse_loss, model_ema):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if args.modal.find('depth') != -1:
        g_depth_losses = AverageMeter()

    model.train()
    end = time.time()
    for step, inputs in enumerate(train_dataloader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        rgb, depth, labels, depthest, n_rgb, n_depth, n_depthest = inputs[
            0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]

        rgb = rgb.to(device, non_blocking=True).float()
        depth = depth.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        depthest = depthest.to(device, non_blocking=True).float()
        n_rgb = n_rgb.to(device, non_blocking=True).float()
        n_depth = n_depth.to(device, non_blocking=True).float()
        n_depthest = n_depthest.to(device, non_blocking=True).float()

        # visulaize_input(rgb, depth, skeleton)
        # assert False

        if args.modal == 'rgb_depth':
            outputs, g_depth_out = model(rgb)
            # depth: [N, T, 1, 224, 224]
            n_l_depth_gt = n_depth.view(
                -1, 1, n_depth.size(-2), n_depth.size(-1))
            g_depth_gt = F.interpolate(
                n_l_depth_gt, size=(56, 56), mode='bilinear')
            g_depth_loss = mse_loss(g_depth_out, g_depth_gt)
            loss = criterion(outputs, labels) + 0.01 * g_depth_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.item(), labels.size(0))
        top1.update(prec1.item(), labels.size(0))
        top5.update(prec5.item(), labels.size(0))
        if args.modal.find('depth') != -1:
            g_depth_losses.update(g_depth_loss.item(), labels.size(0))

        loss.backward()
        optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        batch_time.update(time.time() - end)
        end = time.time()
        logging_str = 'Epoch: [{0}][{1}/{2}], lr: {lr:.5f}, ' \
            'data_time: {data_time.val:.3f} ({data_time.avg:.3f}), ' \
            'batch_time: {batch_time.val:.3f} ({batch_time.avg:.3f}), ' \
            'Top-1: {top1_acc.val:.2f} ({top1_acc.avg:.2f}), ' \
            'Top-5: {top5_acc.val:.2f} ({top5_acc.avg:.2f}), ' \
            'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, step+1, len(train_dataloader), lr=optimizer.param_groups[-1]['lr'],
                data_time=data_time,
                batch_time=batch_time,
                top1_acc=top1,
                top5_acc=top5,
                loss=losses,
            )

        if args.modal.find('depth') != -1:
            logging_str += ', G_Depth_Loss: {g_depth_loss.val:.4f} ({g_depth_loss.avg:.4f})'.format(
                g_depth_loss=g_depth_losses)

        if (step+1) % params['display'] == 0:
            logging.info(logging_str)
            if args.modal.find('depth') != -1:
                ts.save(g_depth_out, args.model_path + 'g_depth_out.jpg')
                ts.save(g_depth_gt, args.model_path + 'g_depth_gt.jpg')

        wandb.log(
            {
                'train_loss': losses.avg,
                'train_top1': top1.avg,
                'train_top5': top5.avg
            }, step=epoch
        )


@torch.no_grad()
def validation(model, val_dataloader, epoch, criterion, is_ema=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    for step, inputs in enumerate(val_dataloader):
        data_time.update(time.time() - end)

        rgb, labels = inputs[0], inputs[2]
        rgb = rgb.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        if args.modal == 'rgb_depth':
            outputs, _ = model(rgb)

        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.item(), labels.size(0))
        top1.update(prec1.item(), labels.size(0))
        top5.update(prec5.item(), labels.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (step + 1) % params['display'] == 0:
            logging.info('Test: [{0}][{1}], '
                         'data_time: {data_time.val:.3f} ({data_time.avg:.3f}), batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                         'loss: {loss.val:.4f} ({loss.avg:.4f}), '
                         'Top-1: {top1_acc.val:.2f} ({top1_acc.avg:.2f}), '
                         'Top-5: {top5_acc.val:.2f} ({top5_acc.avg:.2f})'
                         .format(step+1, len(val_dataloader),
                                 data_time=data_time, batch_time=batch_time,
                                 loss=losses, top1_acc=top1, top5_acc=top5
                                 )
                         )

    logging.info('Testing Results: loss {loss.avg:.5f}, Top-1 {top1.avg:.3f}, Top-5 {top5.avg:.3f}'
                 .format(loss=losses, top1=top1, top5=top5)
                 )

    if is_ema:
        wandb.log(
            {
                'ema_test_loss': losses.avg,
                'ema_test_top1': top1.avg,
                'ema_test_top5': top5.avg
            }, step=epoch
        )
    else:
        wandb.log(
            {
                'test_loss': losses.avg,
                'test_top1': top1.avg,
                'test_top5': top5.avg
            }, step=epoch
        )

    model.train()
    return losses.avg, top1.avg


@torch.no_grad()
def testing(model, val_dataloader, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    for step, inputs in enumerate(tqdm(val_dataloader)):
        rgb, labels = inputs[0], inputs[2]
        rgb = rgb.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        if args.modal == 'rgb_depth':
            outputs, _ = model(rgb)

        loss = criterion(outputs, labels)

        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.item(), labels.size(0))
        top1.update(prec1.item(), labels.size(0))
        top5.update(prec5.item(), labels.size(0))
        batch_time.update(time.time() - end)

    logging.info('loss: {loss:.5f}'.format(loss=losses.avg))
    logging.info('Top-1: {top1_acc:.2f}, Top-5: {top5_acc:.2f}'.format(
        top1_acc=top1.avg,
        top5_acc=top5.avg))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    best_acc = 0.
    best_ema_acc = 0.

    # Seed everything
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(0)

    now = datetime.datetime.now()
    tinfo = "%d-%d-%d-%d-%d-%d" % (now.year, now.month,
                                   now.day, now.hour, now.minute, now.second)
    if args.dataset == 'EgoGesture':
        if args.subset:
            exp_path = "runs/EgoGesture/MTMM_subset/"
        else:
            exp_path = "runs/EgoGesture/MTMM/"
        if args.train_plus_val:
            exp_path = os.path.join(exp_path, "train_plus_val/")
        else:
            exp_path = os.path.join(exp_path, "train/")
    elif args.dataset == 'NvGesture':
        exp_path = "runs/NvGesture/MTMM/"

    exp_name = args.model_name
    args.model_path = exp_path + tinfo + "_" + exp_name + "/"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    logger_file = os.path.join(args.model_path, 'train.log')
    handlers = [logging.FileHandler(logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    logging.info("'''")
    logging.info("Dir: " + args.model_path)
    logging.info("Notes: " + args.notes)
    logging.info("'''")

    wandb.init(
        project=args.dataset,
        entity="peter0512lee",
        name=args.model_name,
        notes=args.notes,
        config=args
    )

    input_mean = [.485, .456, .406]
    input_std = [.229, .224, .225]
    normalize = GroupNormalize(input_mean, input_std)
    scales = [1, .875, .75, .66]

    if args.dataset == 'EgoGesture':
        trans_train = torchvision.transforms.Compose([
            GroupScale([224, 224]),
            GroupMultiScaleCrop([224, 224], scales),
            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.base_model not in [
                                'BNInception', 'InceptionV3'])),
        ])

        temporal_transform_train = torchvision.transforms.Compose([
            TemporalUniformCrop_train(args.clip_len)
        ])

        trans_test = torchvision.transforms.Compose([
            GroupScale([224, 224]),
            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.base_model not in [
                                'BNInception', 'InceptionV3'])),
        ])

        temporal_transform_test = torchvision.transforms.Compose([
            TemporalUniformCrop_val(args.clip_len)
        ])
    elif args.dataset == 'NvGesture':
        trans_train = torchvision.transforms.Compose([
            GroupScale(256),
            GroupMultiScaleCrop(224, scales),
            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.base_model not in [
                                'BNInception', 'InceptionV3'])),
        ])
        temporal_transform_train = torchvision.transforms.Compose([
            TemporalUniformCrop_train(args.clip_len)
        ])
        trans_test = torchvision.transforms.Compose([
            GroupScale(256),
            GroupCenterCrop(224),
            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.base_model not in [
                                'BNInception', 'InceptionV3'])),
        ])
        temporal_transform_test = torchvision.transforms.Compose([
            TemporalUniformCrop_val(args.clip_len)
        ])

    criterion = nn.CrossEntropyLoss().to(device)
    mse_loss = nn.MSELoss().to(device)

    cudnn.benchmark = True
    logging.info("Loading dataset")
    if args.dataset == 'EgoGesture':
        train_dataset = dataset_EgoGesture.dataset_video_MTMM(
            annot_path,
            'train' if not args.train_plus_val else 'train_plus_val',
            spatial_transform=trans_train,
            temporal_transform=temporal_transform_train,
            normalize=normalize,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            num_workers=params['num_workers'],
            shuffle=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_dataset = dataset_EgoGesture.dataset_video_MTMM(
            annot_path,
            'test',
            spatial_transform=trans_test,
            temporal_transform=temporal_transform_test,
            normalize=normalize,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            num_workers=params['num_workers']
        )
    elif args.dataset == 'NvGesture':
        train_dataset = dataset_NvGesture.dataset_video_MTMM(
            annot_path,
            'train',
            spatial_transform=trans_train,
            temporal_transform=temporal_transform_train,
            normalize=normalize,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            num_workers=params['num_workers'],
            shuffle=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_dataset = dataset_NvGesture.dataset_video_MTMM(
            annot_path,
            'test',
            spatial_transform=trans_test,
            temporal_transform=temporal_transform_test,
            normalize=normalize,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            num_workers=params['num_workers']
        )

    logging.info("load model")
    model = TSN_model.TSN(params['num_classes'], args.clip_len, 'RGB',
                          is_shift=args.is_shift,
                          partial_bn=args.npb,
                          base_model=args.base_model,
                          shift_div=args.shift_div,
                          dropout=args.dropout,
                          img_feature_dim=224,
                          pretrain=args.pretrain,
                          consensus_type='avg',
                          fc_lr5=True,
                          modal=args.modal)

    policies = model.get_optim_policies()
    model = model.to(device)
    model_ema = EMAWrapper(model, args.ema_decay, device=device)

    if args.checkpoint_path != '':
        logging.info('loading checkpoint...')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info('First test before training...')
        testing(model, val_dataloader, criterion)

    for param_group in policies:
        param_group['lr'] = args.lr * param_group['lr_mult']
        param_group['weight_decay'] = args.weight_decay * \
            param_group['decay_mult']
    for group in policies:
        logging.info(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    optimizer = optim.SGD(policies, momentum=params['momentum'])

    for epoch in trange(params['epoch_num']):
        train(model, train_dataloader, epoch,
              criterion, optimizer, mse_loss, model_ema)
        latest_checkpoint = os.path.join(
            args.model_path, args.model_name + "_latest_checkpoint" + ".pth.tar")
        utils.save_checkpoint(model, optimizer, latest_checkpoint)
        val_loss, val_acc = validation(
            model, val_dataloader, epoch, criterion)
        val_loss, ema_val_acc = validation(
            model_ema.model, val_dataloader, epoch, criterion, True)
        if val_acc > best_acc:
            best_checkpoint = os.path.join(
                args.model_path, args.model_name + "_best_checkpoint" + ".pth.tar")
            utils.save_checkpoint(model, optimizer, best_checkpoint)
            best_acc = val_acc
        if ema_val_acc > best_ema_acc:
            best_checkpoint = os.path.join(
                args.model_path, args.model_name + "_ema_best_checkpoint" + ".pth.tar")
            utils.save_checkpoint(
                model_ema.model, optimizer, best_checkpoint)
            best_ema_acc = ema_val_acc
        logging.info('Best Top-1: {:.2f}'.format(best_acc))
        logging.info('Best EMA Top-1: {:.2f}'.format(best_ema_acc))
        wandb.log({'best_prec1': best_acc}, step=epoch)
        wandb.log({'best_ema_prec1': best_ema_acc}, step=epoch)
        utils.adjust_learning_rate(
            params['learning_rate'], optimizer, epoch, args.lr_steps)


if __name__ == '__main__':
    main()
