import argparse
import logging
import os
import time
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import utils as utils
from data import dataset_EgoGesture, dataset_NvGesture
from models.models_SD_actionnet import TSN
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
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--train_plus_val', action="store_true")

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

    # kd parameter
    parser.add_argument('--temperature', default=3, type=int,
                        help='temperature to smooth the logits')
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='weight of kd loss')
    parser.add_argument('--beta', default=1e-6, type=float,
                        help='weight of feature loss')

    args = parser.parse_args()
    return args


args = parse_opts()

params = dict()
if args.dataset == 'EgoGesture':
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


def kd_loss_function(output, target_output, args):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / args.temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd


def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()


def train(model, train_dataloader, epoch, criterion, optimizer, model_ema):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    middle1_losses = AverageMeter()
    middle2_losses = AverageMeter()
    middle3_losses = AverageMeter()
    losses1_kd = AverageMeter()
    losses2_kd = AverageMeter()
    losses3_kd = AverageMeter()
    feature_losses_1 = AverageMeter()
    feature_losses_2 = AverageMeter()
    feature_losses_3 = AverageMeter()
    total_losses = AverageMeter()
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()
    middle3_top1 = AverageMeter()

    model.train()
    end = time.time()
    for step, inputs in enumerate(train_dataloader):
        data_time.update(time.time() - end)

        rgb, labels = inputs[0], inputs[1]
        rgb = rgb.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        output, middle_output1, middle_output2, middle_output3, \
            final_fea, middle1_fea, middle2_fea, middle3_fea = model(rgb)

        # cross entropy loss
        loss = criterion(output, labels)
        losses.update(loss.item(), labels.size(0))
        middle1_loss = criterion(middle_output1, labels)
        middle1_losses.update(middle1_loss.item(), labels.size(0))
        middle2_loss = criterion(middle_output2, labels)
        middle2_losses.update(middle2_loss.item(), labels.size(0))
        middle3_loss = criterion(middle_output3, labels)
        middle3_losses.update(middle3_loss.item(), labels.size(0))

        # KD loss
        temp4 = output / args.temperature
        temp4 = torch.softmax(temp4, dim=1)
        loss1by4 = kd_loss_function(
            middle_output1, temp4.detach(), args) * (args.temperature**2)
        losses1_kd.update(loss1by4.item(), labels.size(0))
        loss2by4 = kd_loss_function(
            middle_output2, temp4.detach(), args) * (args.temperature**2)
        losses2_kd.update(loss2by4.item(), labels.size(0))
        loss3by4 = kd_loss_function(
            middle_output3, temp4.detach(), args) * (args.temperature**2)
        losses3_kd.update(loss3by4.item(), labels.size(0))

        # L2 loss
        feature_loss_1 = feature_loss_function(
            middle1_fea, final_fea.detach())
        feature_losses_1.update(feature_loss_1.item(), labels.size(0))
        feature_loss_2 = feature_loss_function(
            middle2_fea, final_fea.detach())
        feature_losses_2.update(feature_loss_2.item(), labels.size(0))
        feature_loss_3 = feature_loss_function(
            middle3_fea, final_fea.detach())
        feature_losses_3.update(feature_loss_3.item(), labels.size(0))

        # total loss
        total_loss = (1 - args.alpha) * (loss + middle1_loss + middle2_loss + middle3_loss) + \
            args.alpha * (loss1by4 + loss2by4 + loss3by4) + \
            args.beta * (feature_loss_1 + feature_loss_2 + feature_loss_3)
        total_losses.update(total_loss.item(), labels.size(0))

        # Accuracy
        prec1 = accuracy(output.data, labels, topk=(1,))
        top1.update(prec1[0].item(), labels.size(0))
        middle1_prec1 = accuracy(middle_output1.data, labels, topk=(1,))
        middle1_top1.update(middle1_prec1[0].item(), labels.size(0))
        middle2_prec1 = accuracy(middle_output2.data, labels, topk=(1,))
        middle2_top1.update(middle2_prec1[0].item(), labels.size(0))
        middle3_prec1 = accuracy(middle_output3.data, labels, topk=(1,))
        middle3_top1.update(middle3_prec1[0].item(), labels.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        batch_time.update(time.time() - end)
        end = time.time()
        if (step+1) % params['display'] == 0:
            logging.info('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}, '
                         'Data_time: {data_time.val:.3f} ({data_time.avg:.3f}), batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                         'Loss: {loss.val:.4f} ({loss.avg:.4f}), '
                         'Top-1: {top1_acc.val:.2f} ({top1_acc.avg:.2f}), '
                         'Middle1@1: {middle1_top1.val:.2f} ({middle1_top1.avg:.2f}), '
                         'Middle2@1: {middle2_top1.val:.2f} ({middle2_top1.avg:.2f}), '
                         'Middle3@1: {middle3_top1.val:.2f} ({middle3_top1.avg:.2f}) '
                         .format(epoch, step+1, len(train_dataloader),
                                 lr=optimizer.param_groups[2]['lr'],
                                 data_time=data_time, batch_time=batch_time,
                                 loss=total_losses, top1_acc=top1, middle1_top1=middle1_top1,
                                 middle2_top1=middle2_top1, middle3_top1=middle3_top1
                                 )
                         )

        wandb.log(
            {
                'train_loss': total_losses.avg,
                'train_top1': top1.avg,
                'train_middle1_loss': middle1_losses.avg,
                'train_middle1_top1': middle1_top1.avg,
                'train_middle2_loss': middle2_losses.avg,
                'train_middle2_top1': middle2_top1.avg,
                'train_middle3_loss': middle3_losses.avg,
                'train_middle3_top1': middle3_top1.avg,
            }, step=epoch
        )


@torch.no_grad()
def validation(model, val_dataloader, epoch, criterion, is_ema=False):
    losses = AverageMeter()
    middle1_losses = AverageMeter()
    middle2_losses = AverageMeter()
    middle3_losses = AverageMeter()
    top1 = AverageMeter()
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()
    middle3_top1 = AverageMeter()
    model.eval()

    for step, inputs in enumerate(val_dataloader):
        rgb, labels = inputs[0], inputs[1]
        rgb = rgb.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        output, middle_output1, middle_output2, middle_output3, \
            _, _, _, _ = model(rgb)

        # cross entropy loss
        loss = criterion(output, labels)
        losses.update(loss.item(), labels.size(0))
        middle1_loss = criterion(middle_output1, labels)
        middle1_losses.update(middle1_loss.item(), labels.size(0))
        middle2_loss = criterion(middle_output2, labels)
        middle2_losses.update(middle2_loss.item(), labels.size(0))
        middle3_loss = criterion(middle_output3, labels)
        middle3_losses.update(middle3_loss.item(), labels.size(0))

        # Accuracy
        prec1 = accuracy(output.data, labels, topk=(1,))
        top1.update(prec1[0].item(), labels.size(0))
        middle1_prec1 = accuracy(middle_output1.data, labels, topk=(1,))
        middle1_top1.update(middle1_prec1[0].item(), labels.size(0))
        middle2_prec1 = accuracy(middle_output2.data, labels, topk=(1,))
        middle2_top1.update(middle2_prec1[0].item(), labels.size(0))
        middle3_prec1 = accuracy(middle_output3.data, labels, topk=(1,))
        middle3_top1.update(middle3_prec1[0].item(), labels.size(0))

        if (step + 1) % params['display'] == 0:
            logging.info('Test: [{0}][{1}], '
                         'Loss: {loss.val:.4f} ({loss.avg:.4f}), '
                         'Top-1: {top1_acc.val:.2f} ({top1_acc.avg:.2f}), '
                         'Middle1@1: {middle1_top1.val:.2f} ({middle1_top1.avg:.2f}), '
                         'Middle2@1: {middle2_top1.val:.2f} ({middle2_top1.avg:.2f}), '
                         'Middle3@1: {middle3_top1.val:.2f} ({middle3_top1.avg:.2f}) '
                         .format(step+1, len(val_dataloader),
                                 loss=losses,
                                 top1_acc=top1,
                                 middle1_top1=middle1_top1,
                                 middle2_top1=middle2_top1,
                                 middle3_top1=middle3_top1
                                 )
                         )

    logging.info('Testing Results: loss {loss.avg:.5f}, Top-1 {top1.avg:.3f}'
                 .format(loss=losses, top1=top1))
    if is_ema:
        wandb.log(
            {
                'ema_test_loss': losses.avg,
                'ema_test_top1': top1.avg,
                'ema_test_middle1_loss': middle1_losses.avg,
                'ema_test_middle1_top1': middle1_top1.avg,
                'ema_test_middle2_loss': middle2_losses.avg,
                'ema_test_middle2_top1': middle2_top1.avg,
                'ema_test_middle3_loss': middle3_losses.avg,
                'ema_test_middle3_top1': middle3_top1.avg,
            }, step=epoch
        )
    else:
        wandb.log(
            {
                'test_loss': losses.avg,
                'test_top1': top1.avg,
                'test_middle1_loss': middle1_losses.avg,
                'test_middle1_top1': middle1_top1.avg,
                'test_middle2_loss': middle2_losses.avg,
                'test_middle2_top1': middle2_top1.avg,
                'test_middle3_loss': middle3_losses.avg,
                'test_middle3_top1': middle3_top1.avg,
            }, step=epoch
        )

    model.train()
    return top1.avg, middle1_top1.avg, middle2_top1.avg, middle3_top1.avg


@torch.no_grad()
def testing(model, val_dataloader, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    for step, inputs in enumerate(tqdm(val_dataloader)):
        rgb, labels = inputs[0], inputs[1]
        rgb = rgb.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        output, _, _, _, _, _, _, _ = model(rgb)
        loss = criterion(output, labels)

        prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))
        losses.update(loss.item(), labels.size(0))
        top1.update(prec1.item(), labels.size(0))
        top5.update(prec5.item(), labels.size(0))

    logging.info('loss: {loss:.5f}'.format(loss=losses.avg))
    logging.info(
        'Top-1: {top1_acc:.2f}, Top-5: {top5_acc:.2f}'.format(top1_acc=top1.avg, top5_acc=top5.avg))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    best_acc = 0.
    best_ema_acc = 0.

    # Seed everything
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(0)

    if args.dataset == 'EgoGesture':
        exp_path = "runs/EgoGesture/SD/"
        if args.train_plus_val:
            exp_path = os.path.join(exp_path, "train_plus_val/")
        else:
            exp_path = os.path.join(exp_path, "train/")
    elif args.dataset == 'NvGesture':
        exp_path = "runs/NvGesture/SD/"

    args.model_name = "ACTION-Net_SD"
    args.model_path = exp_path + args.model_name + "/"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    logger_file = os.path.join(args.model_path, 'train.log')
    handlers = [logging.FileHandler(logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    logging.info(args.model_path)

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
            normalize,
        ])

        temporal_transform_train = torchvision.transforms.Compose([
            TemporalUniformCrop_train(args.clip_len)
        ])

        trans_test = torchvision.transforms.Compose([
            GroupScale([224, 224]),
            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.base_model not in [
                                'BNInception', 'InceptionV3'])),
            normalize,
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
            normalize,
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
            normalize,
        ])
        temporal_transform_test = torchvision.transforms.Compose([
            TemporalUniformCrop_val(args.clip_len)
        ])

    criterion = nn.CrossEntropyLoss().to(device)

    cudnn.benchmark = True
    logging.info("Loading dataset")
    if args.dataset == 'EgoGesture':
        train_dataset = dataset_EgoGesture.dataset_video_distill(
            annot_path,
            'train' if not args.train_plus_val else 'train_plus_val',
            spatial_transform=trans_train,
            temporal_transform=temporal_transform_train,
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
        val_dataset = dataset_EgoGesture.dataset_video_distill(
            annot_path,
            'test',
            spatial_transform=trans_test,
            temporal_transform=temporal_transform_test,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            num_workers=params['num_workers']
        )
    elif args.dataset == 'NvGesture':
        train_dataset = dataset_NvGesture.dataset_video_distill(
            annot_path,
            'train',
            spatial_transform=trans_train,
            temporal_transform=temporal_transform_train,
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
        val_dataset = dataset_NvGesture.dataset_video_distill(
            annot_path,
            'test',
            spatial_transform=trans_test,
            temporal_transform=temporal_transform_test,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            num_workers=params['num_workers']
        )

    logging.info("load model")
    model = TSN(params['num_classes'], args.clip_len, 'RGB',
                is_shift=args.is_shift,
                partial_bn=args.npb,
                base_model=args.base_model,
                shift_div=args.shift_div,
                dropout=0,
                img_feature_dim=224,
                pretrain=args.pretrain,
                consensus_type='avg',
                fc_lr5=True,)

    if args.resume:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    policies = model.get_optim_policies()
    model = model.to(device)
    model_ema = EMAWrapper(model, 0.9999, device=device)

    if args.checkpoint_path != '':
        logging.info('First test before training')
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
        train(model, train_dataloader, epoch, criterion, optimizer, model_ema)
        if epoch % 1 == 0:
            latest_checkpoint = os.path.join(
                args.model_path, args.model_name + "_latest_checkpoint" + ".pth.tar")
            utils.save_checkpoint(model, optimizer, latest_checkpoint)
            val_acc, val_middle1_acc, val_middle2_acc, val_middle3_acc = validation(
                model, val_dataloader, epoch, criterion)
            ema_val_acc, ema_val_middle1_acc, ema_val_middle2_acc, ema_val_middle3_acc = validation(
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
            logging.info('Best Top-1: {:.2f}'.format(best_acc))
            logging.info('Best EMA Top-1: {:.2f}'.format(best_ema_acc))
            wandb.log({'best_prec1': best_acc}, step=epoch)
            wandb.log({'best_ema_prec1': best_ema_acc}, step=epoch)
        utils.adjust_learning_rate(
            params['learning_rate'], optimizer, epoch, args.lr_steps)


if __name__ == '__main__':
    main()
