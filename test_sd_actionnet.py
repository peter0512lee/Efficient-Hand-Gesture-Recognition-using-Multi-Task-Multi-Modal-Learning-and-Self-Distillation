import argparse
import logging
import os
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import wandb
from PIL import ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils as utils
from data import dataset_EgoGesture, dataset_NvGesture
from models.models_SD_actionnet import TSN
from models.spatial_transforms import *
from models.temporal_transforms import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore")

os.environ['WANDB_MODE'] = 'disabled'


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='0')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--single_clip_test', action='store_true')
    parser.add_argument('--multiple_clip_test', action='store_true')

    # args for dataloader
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--clip_len', type=int, default=8)

    # args for preprocessing
    parser.add_argument('--shift_div', default=8, type=int)
    parser.add_argument('--is_shift', default=True, type=bool)
    parser.add_argument('--base_model', default='resnet50', type=str)
    parser.add_argument('--dataset', default='EgoGesture', type=str)

    # args for testing
    parser.add_argument('--test_crops', type=int, default=1)
    parser.add_argument('--scale_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--clip_num', type=int, default=10)

    args = parser.parse_args()
    return args


args = parse_opts()

params = dict()
if args.dataset == 'EgoGesture':
    params['num_classes'] = 83
elif args.dataset == 'NvGesture':
    params['num_classes'] = 25

annot_path = 'data/{}_annotation'.format(args.dataset)
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
device = 'cuda:0'


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


@torch.no_grad()
def inference(model, val_dataloader):
    top1 = AverageMeter()
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()
    middle3_top1 = AverageMeter()
    model.eval()

    for step, inputs in enumerate(tqdm(val_dataloader)):
        rgb, labels = inputs[0], inputs[1]
        rgb = rgb.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        nb, n_clip, nt, nc, h, w = rgb.size()
        # n_clip * nb (1) * crops, T, C, H, W
        rgb = rgb.view(-1, nt//args.test_crops, nc, h, w)
        outputs, middle_output1, middle_output2, middle_output3, \
            _, _, _, _ = model(rgb)

        outputs = outputs.view(nb, n_clip*args.test_crops, -1)
        outputs = F.softmax(outputs, 2)
        middle_output1 = middle_output1.view(nb, n_clip*args.test_crops, -1)
        middle_output1 = F.softmax(middle_output1, 2)
        middle_output2 = middle_output2.view(nb, n_clip*args.test_crops, -1)
        middle_output2 = F.softmax(middle_output2, 2)
        middle_output3 = middle_output3.view(nb, n_clip*args.test_crops, -1)
        middle_output3 = F.softmax(middle_output3, 2)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data.mean(1), labels, topk=(1, ))
        top1.update(prec1[0].item(), labels.size(0))
        middle1_prec1 = accuracy(
            middle_output1.data.mean(1), labels, topk=(1,))
        middle1_top1.update(middle1_prec1[0].item(), labels.size(0))
        middle2_prec1 = accuracy(
            middle_output2.data.mean(1), labels, topk=(1,))
        middle2_top1.update(middle2_prec1[0].item(), labels.size(0))
        middle3_prec1 = accuracy(
            middle_output3.data.mean(1), labels, topk=(1,))
        middle3_top1.update(middle3_prec1[0].item(), labels.size(0))

    logging.info('Top-1: {top1_acc:.2f}, '
                 'Middle1@1: {middle1_top1:.2f}, '
                 'Middle2@1: {middle2_top1:.2f}, '
                 'Middle3@1: {middle3_top1:.2f} '
                 .format(top1_acc=top1.avg,
                         middle1_top1=middle1_top1.avg,
                         middle2_top1=middle2_top1.avg,
                         middle3_top1=middle3_top1.avg
                         )
                 )

    wandb.log(
        {
            'Top-1': top1.avg,
            'Middle1_top1': middle1_top1.avg,
            'Middle2_top1': middle2_top1.avg,
            'Middle3_top1': middle3_top1.avg,
        }
    )


if __name__ == '__main__':

    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    exp_path = args.checkpoint_path.split(
        '/')[0] + '/' + args.checkpoint_path.split(
        '/')[1] + '/' + args.checkpoint_path.split(
        '/')[2] + '/' + args.checkpoint_path.split(
        '/')[3] + '/' + args.checkpoint_path.split('/')[4]

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    logger_file = os.path.join(exp_path, 'test.log')
    handlers = [logging.FileHandler(logger_file, mode='a'),  # ! mode='w' will overwrite the log file
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)
    logging.info(args.note + '_test_crops' + str(args.test_crops) + '_clip_num' + str(
        args.clip_num) + '_scale_size' + str(args.scale_size) + '_crop_size' + str(args.crop_size))

    wandb.init(
        project=args.dataset + '_Inference',
        name=args.checkpoint_path.split('/')[4],
        notes=args.note,
        config=args
    )

    if args.dataset == 'EgoGesture':
        cropping = torchvision.transforms.Compose([
            GroupScale([args.crop_size, args.crop_size])
        ])
    else:
        if args.test_crops == 1:
            cropping = torchvision.transforms.Compose([
                GroupScale(args.scale_size),
                GroupCenterCrop(args.crop_size)
            ])
        elif args.test_crops == 3:
            cropping = torchvision.transforms.Compose([
                GroupFullResSample(args.crop_size, args.scale_size, flip=False)
            ])
        elif args.test_crops == 5:
            cropping = torchvision.transforms.Compose([
                GroupOverSample(args.crop_size, args.scale_size, flip=False)
            ])

    input_mean = [.485, .456, .406]
    input_std = [.229, .224, .225]
    normalize = GroupNormalize(input_mean, input_std)

    spatial_transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
        ToTorchFormatTensor(div=(args.base_model not in [
                            'BNInception', 'InceptionV3'])),
        normalize
    ])

    # for mulitple clip test, use random sampling;
    # for single clip test, use middle sampling
    if args.single_clip_test:
        temporal_transform = torchvision.transforms.Compose([
            TemporalUniformCrop_val(args.clip_len)
        ])
    if args.multiple_clip_test:
        temporal_transform = torchvision.transforms.Compose([
            TemporalUniformCrop_train(args.clip_len)
        ])

    cudnn.benchmark = True
    model = TSN(params['num_classes'], args.clip_len, 'RGB',
                is_shift=args.is_shift,
                base_model=args.base_model,
                shift_div=args.shift_div,
                img_feature_dim=args.crop_size,
                consensus_type='avg',
                dropout=0,
                fc_lr5=True)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(device)

    if args.dataset == 'EgoGesture':
        val_dataset = dataset_EgoGesture.dataset_video_inference(
            annot_path,
            'test',
            clip_num=args.clip_num,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    elif args.dataset == 'NvGesture':
        val_dataset = dataset_NvGesture.dataset_video_inference(
            annot_path,
            'test',
            clip_num=args.clip_num,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    inference(model, val_dataloader)
