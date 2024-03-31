import argparse
import logging
import os
import time
import warnings

import cv2
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import wandb
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils as utils
from data import dataset_EgoGesture
from models import models_mtask as TSN_model
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
    parser.add_argument('--modal', type=str, default='rgb')

    # args for dataloader
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--clip_len', type=int, default=8)

    # args for preprocessing
    parser.add_argument('--shift_div', default=8, type=int)
    parser.add_argument('--is_shift', default=True, type=bool)
    parser.add_argument('--base_model', default='resnet50', type=str)
    parser.add_argument('--dataset', default='EgoGesture', type=str)
    parser.add_argument('--dropout', default=0.5, type=float)

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


def export_prediction_video(image_paths, folder_name, pred, label, classIndAll):
    # Get the dimensions of the first image to use as the video frame size
    first_image = cv2.imread(image_paths[0])
    frame_size = (first_image.shape[1]*2, first_image.shape[0]*2)

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_write_path = os.path.join(
        save_path, folder_name, folder_name) + '.mp4'
    video_writer = cv2.VideoWriter(video_write_path, fourcc, 30, frame_size)

    logging.info(folder_name)
    logging.info(f'Pred: {classIndAll[pred][1]}')
    logging.info(f'Label: {classIndAll[label][1]}')
    logging.info('-' * 40)

    # Loop through the list of image paths and write each image to the video file
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, frame_size)
        pred_text = 'pred: ' + classIndAll[pred][1]
        label_text = 'label: ' + classIndAll[label][1]
        cv2.putText(image, pred_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, label_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        video_writer.write(image)

    # Release the video writer and close the video file
    video_writer.release()
    cv2.destroyAllWindows()


def export_reconstructed_depth(folder_name, local_depth, local_depth_gt, global_depth, global_depth_gt):
    depth_save_path = os.path.join(save_path, folder_name, 'depth')
    if not os.path.exists(depth_save_path):
        os.makedirs(depth_save_path)

    for i in range(local_depth.shape[0]):

        l_depth_loss = F.mse_loss(local_depth[i], local_depth_gt[i])
        g_depth_loss = F.mse_loss(global_depth[i], global_depth_gt[i])

        l_depth = local_depth[i].squeeze(0).cpu().numpy()
        l_depth_gt = local_depth_gt[i].squeeze(0).cpu().numpy()
        g_depth = global_depth[i].squeeze(0).cpu().numpy()
        g_depth_gt = global_depth_gt[i].squeeze(0).cpu().numpy()

        l_depth = l_depth * 255
        l_depth_gt = l_depth_gt * 255
        g_depth = g_depth * 255
        g_depth_gt = g_depth_gt * 255

        l_depth_array = np.zeros((l_depth.shape[0], l_depth.shape[1] * 2))
        l_depth_array[:, :l_depth.shape[1]] = l_depth
        l_depth_array[:, l_depth.shape[1]:] = l_depth_gt
        plt.figure(figsize=(4, 4))
        plt.imshow(l_depth_array, cmap='gray')
        plt.title('local depth loss: {:.4f}'.format(l_depth_loss))
        plt.savefig(os.path.join(depth_save_path,
                    'local_depth_{}.jpg'.format(i)))
        plt.close()
        # plt.imsave(os.path.join(depth_save_path,
        #            'local_depth_{}.jpg'.format(i)), l_depth_array, cmap='gray')
        g_depth_array = np.zeros((g_depth.shape[0], g_depth.shape[1] * 2))
        g_depth_array[:, :g_depth.shape[1]] = g_depth
        g_depth_array[:, g_depth.shape[1]:] = g_depth_gt
        plt.figure(figsize=(4, 4))
        plt.imshow(g_depth_array, cmap='gray')
        plt.title('global depth loss: {:.4f}'.format(g_depth_loss))
        plt.savefig(os.path.join(depth_save_path,
                    'global_depth_{}.jpg'.format(i)))
        # plt.imsave(os.path.join(depth_save_path,
        #            'global_depth_{}.jpg'.format(i)), g_depth_array, cmap='gray')
        plt.close()
    # assert False


def export_reconstructed_skeleton(folder_name, local_skeleton, local_skeleton_gt, global_skeleton, global_skeleton_gt):
    skeleton_save_path = os.path.join(save_path, folder_name, 'skeleton')
    if not os.path.exists(skeleton_save_path):
        os.makedirs(skeleton_save_path)

    for i in range(local_skeleton.shape[0]):
        l_skeleton = local_skeleton[i].squeeze(0).cpu().numpy()
        l_skeleton_gt = local_skeleton_gt[i].squeeze(0).cpu().numpy()
        g_skeleton = global_skeleton[i].squeeze(0).cpu().numpy()
        g_skeleton_gt = global_skeleton_gt[i].squeeze(0).cpu().numpy()

        l_skeleton = np.sum(l_skeleton, axis=0)
        l_skeleton -= l_skeleton.min()
        l_skeleton /= l_skeleton.max()
        l_skeleton_gt = np.sum(l_skeleton_gt, axis=0)
        l_skeleton_gt -= l_skeleton_gt.min()
        l_skeleton_gt /= l_skeleton_gt.max()

        g_skeleton = np.sum(g_skeleton, axis=0)
        g_skeleton -= g_skeleton.min()
        g_skeleton /= g_skeleton.max()
        g_skeleton_gt = np.sum(g_skeleton_gt, axis=0)
        g_skeleton_gt -= g_skeleton_gt.min()
        g_skeleton_gt /= g_skeleton_gt.max()

        l_skeleton_array = np.zeros(
            (l_skeleton.shape[0], l_skeleton.shape[1] * 2))
        l_skeleton_array[:, :l_skeleton.shape[1]] = l_skeleton
        l_skeleton_array[:, l_skeleton.shape[1]:] = l_skeleton_gt

        g_skeleton_array = np.zeros(
            (g_skeleton.shape[0], g_skeleton.shape[1] * 2))
        g_skeleton_array[:, :g_skeleton.shape[1]] = g_skeleton
        g_skeleton_array[:, g_skeleton.shape[1]:] = g_skeleton_gt

        plt.imsave(os.path.join(skeleton_save_path,
                   'l_skeleton_{}.jpg'.format(i)), l_skeleton_array, cmap='gray')
        plt.imsave(os.path.join(skeleton_save_path,
                   'g_skeleton_{}.jpg'.format(i)), g_skeleton_array, cmap='gray')

    # assert False


def case_study(model, val_dataloader, classIndAll):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    count = 0
    model.eval()
    end = time.time()
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(val_dataloader)):
            data_time.update(time.time() - end)
            rgb, depth, labels, text, rgb_name, depth_est = inputs[
                0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
            rgb = rgb.to(device, non_blocking=True).float()
            depth = depth.to(device, non_blocking=True).float()
            depth_est = depth_est.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()
            text = text.to(device, non_blocking=True).float()
            # skeleton = skeleton.to(device, non_blocking=True).float()

            nb, n_clip, nt, nc, h, w = rgb.size()
            # n_clip * nb (1) * crops, T, C, H, W
            rgb = rgb.view(-1, nt//args.test_crops, nc, h, w)

            if args.modal == 'rgb':
                outputs = model(rgb)
            elif args.modal == 'rgb_depth':
                outputs, local_mask, global_mask = model(rgb)
                # depth: [N, T, 1, 224, 224]
                local_mask_gt = depth.view(-1, 1,
                                           depth.size(-2), depth.size(-1))
                global_mask_gt = F.interpolate(
                    local_mask_gt, size=(56, 56), mode='bilinear')
            # elif args.modal == 'rgb_skeleton':
            #     outputs, local_skeleton, global_skeleton = model(rgb)
            #     # skeleton: [N, T, 42, 224, 224]
            #     local_skeleton_gt = skeleton.view(-1, 42,
            #                                       skeleton.size(-2), skeleton.size(-1))
            #     global_skeleton_gt = F.interpolate(
            #         local_skeleton_gt, size=(56, 56), mode='bilinear')
            # elif args.modal == 'rgb_text':
            #     outputs, text_encoded = model(rgb)
            # elif args.modal == 'rgb_depth_text':
            #     outputs, local_mask, global_mask, text_encoded = model(rgb)
            #     local_mask_gt = depth.view(-1, 1,
            #                                depth.size(-2), depth.size(-1))
            #     global_mask_gt = F.interpolate(
            #         local_mask_gt, size=(56, 56), mode='bilinear')
            # elif args.modal == 'rgb_depth_skeleton':
            #     outputs, local_mask, global_mask, local_skeleton, global_skeleton = model(
            #         rgb)
            #     local_mask_gt = depth.view(-1, 1,
            #                                depth.size(-2), depth.size(-1))
            #     global_mask_gt = F.interpolate(
            #         local_mask_gt, size=(56, 56), mode='bilinear')
            #     local_skeleton_gt = skeleton.view(-1, 42,
            #                                       skeleton.size(-2), skeleton.size(-1))
            #     global_skeleton_gt = F.interpolate(
            #         local_skeleton_gt, size=(56, 56), mode='bilinear')
            # elif args.modal == 'rgb_skeleton_text':
            #     outputs, local_skeleton, global_skeleton, text_encoded = model(
            #         rgb)
            #     local_skeleton_gt = skeleton.view(-1, 42,
            #                                       skeleton.size(-2), skeleton.size(-1))
            #     global_skeleton_gt = F.interpolate(
            #         local_skeleton_gt, size=(56, 56), mode='bilinear')
            # elif args.modal == 'rgb_depth_skeleton_text':
            #     outputs, local_mask, global_mask, local_skeleton, global_skeleton, text_encoded = model(
            #         rgb)
            #     local_mask_gt = depth.view(-1, 1,
            #                                depth.size(-2), depth.size(-1))
            #     global_mask_gt = F.interpolate(
            #         local_mask_gt, size=(56, 56), mode='bilinear')
            #     local_skeleton_gt = skeleton.view(-1, 42,
            #                                       skeleton.size(-2), skeleton.size(-1))
            #     global_skeleton_gt = F.interpolate(
            #         local_skeleton_gt, size=(56, 56), mode='bilinear')

            outputs = outputs.view(nb, n_clip*args.test_crops, -1)
            outputs = F.softmax(outputs, 2)

            # Check wrong prediction
            if outputs.data.mean(1).argmax(1) != labels:
                count += 1
                logging.info(f'Wrong prediction: {count}')

            pred = outputs.data.mean(1).argmax(1).item()
            label = labels.item()
            rgb_name = [i[0] for i in rgb_name]
            folder_name = rgb_name[0].split('/')
            clip_name = folder_name[10].split('.')[0]
            # Subject_Scene_rgb_000001_pred_label
            folder_name = folder_name[6] + '_' + folder_name[7] + '_' + folder_name[9] + '_' + clip_name + \
                '_' + str(pred) + '_' + str(label)
            if not os.path.exists(os.path.join(save_path, folder_name)):
                os.makedirs(os.path.join(save_path, folder_name))

            export_prediction_video(
                rgb_name, folder_name, pred, label, classIndAll)
            if args.modal.find('depth') != -1:
                export_reconstructed_depth(
                    folder_name, local_mask, local_mask_gt, global_mask, global_mask_gt)
            # if args.modal.find('skeleton') != -1:
            #     export_reconstructed_skeleton(
            #         folder_name, local_skeleton, local_skeleton_gt, global_skeleton, global_skeleton_gt)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data.mean(1), labels, topk=(1, 5))
            top1.update(prec1.item(), labels.size(0))
            top5.update(prec5.item(), labels.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

        logging.info('Top-1: {top1_acc:.2f}, ' 'Top-5: {top5_acc:.2f}'.format(
            top1_acc=top1.avg,
            top5_acc=top5.avg)
        )
        logging.info(f'All Wrong prediction: {count}')

        wandb.log({'Top-1': top1.avg, 'Top-5': top5.avg})


if __name__ == '__main__':

    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    root_path = 'D:/dev/ACTION-Net/'
    exp_path = "/".join(args.checkpoint_path.split('/')[:5])
    save_path = root_path + exp_path + '/' + 'case_study_' + \
        args.note + '_' + str(args.crop_size) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger_file = os.path.join(
        save_path, 'case_study_' + args.note + '_' + str(args.crop_size) + '.log')
    handlers = [logging.FileHandler(logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)
    logging.info('Logging to {}'.format(logger_file))

    wandb.init(
        project='EgoGesture_Inference',
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
    model = TSN_model.TSN(params['num_classes'], args.clip_len, 'RGB',
                          is_shift=args.is_shift,
                          base_model=args.base_model,
                          shift_div=args.shift_div,
                          img_feature_dim=args.crop_size,
                          consensus_type='avg',
                          fc_lr5=True,
                          dropout=args.dropout,
                          modal=args.modal)

    checkpoint = torch.load(args.checkpoint_path)
    logging.info("load checkpoint {}".format(args.checkpoint_path))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.to(device)

    if args.dataset == 'EgoGesture':
        val_dataset = dataset_EgoGesture.dataset_video_case_study(
            annot_path,
            'test',
            clip_num=args.clip_num,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            normalize=normalize
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    with open('./data/EgoGesture_annotation/classIndAll.txt') as f:
        classIndAll = f.readlines()
        classIndAll = [i.strip().split(' ') for i in classIndAll]

    case_study(model, val_dataloader, classIndAll)
