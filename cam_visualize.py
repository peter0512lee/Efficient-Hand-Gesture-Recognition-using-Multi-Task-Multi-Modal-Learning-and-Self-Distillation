import argparse
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from PIL import ImageFile
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils as utils
from data import dataset_EgoGesture
from models import models as TSN_model
from models.spatial_transforms import *
from models.temporal_transforms import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    parser.add_argument('--num_workers', type=int, default=1)
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
annot_path = 'data/{}_annotation'.format(args.dataset)
device = 'cuda:0'


def main(model, val_dataloader):
    model.eval()

    target_layers = [model.base_model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    root_path = 'D:/dev/ACTION-Net/'
    exp_path = args.checkpoint_path.split(
        '/')[0] + '/' + args.checkpoint_path.split('/')[1] + '/' + args.checkpoint_path.split('/')[2]
    save_path = root_path + exp_path + '/' + 'case_study_' + \
        args.note + '_' + str(args.crop_size) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for step, inputs in enumerate(tqdm(val_dataloader)):
        rgb, labels, rgb_name = inputs[0], inputs[2], inputs[5]
        rgb = rgb.to(device, non_blocking=True).float()
        nb, n_clip, nt, nc, h, w = rgb.size()
        rgb = rgb.view(-1, nt//args.test_crops, nc, h, w)

        # Make model output
        outputs = model(rgb)
        outputs = outputs.view(nb, n_clip*args.test_crops, -1)
        outputs = F.softmax(outputs, 2)
        pred = outputs.data.mean(1).argmax(1).item()
        label = labels.item()

        targets = [ClassifierOutputTarget(label)]
        rgb_name = [i[0] for i in rgb_name]
        folder_name = rgb_name[0].split('/')
        clip_name = folder_name[10].split('.')[0]
        # Subject_Scene_rgb_000001_pred_label
        folder_name = folder_name[6] + '_' + folder_name[7] + '_' + folder_name[9] + '_' + clip_name + \
            '_' + str(pred) + '_' + str(label) + '/' + 'cam' + '/'
        if not os.path.exists(save_path + folder_name):
            os.makedirs(save_path + folder_name)
        video_writer = cv2.VideoWriter(
            save_path + folder_name + 'cam.mp4', fourcc, 8.0, (224, 224))
        for i in range(args.clip_len):
            img = rgb[0, i, :, :, :]
            img = (img - img.min()) / (img.max() - img.min())
            img = img.permute(1, 2, 0).cpu().numpy()
            grayscale_cam = cam(input_tensor=rgb, targets=targets)
            cam_image = show_cam_on_image(img, grayscale_cam[i], use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            # img save to folder
            cv2.imwrite(save_path + folder_name + 'cam_' +
                        str(i) + '.jpg', cam_image)
            # write to video
            video_writer.write(cam_image)
        # Release the video writer and close the video file
        video_writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
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
    model = TSN_model.TSN(83, args.clip_len, 'RGB',
                          is_shift=args.is_shift,
                          base_model=args.base_model,
                          shift_div=args.shift_div,
                          img_feature_dim=args.crop_size,
                          consensus_type='avg',
                          fc_lr5=True)

    checkpoint = torch.load(args.checkpoint_path)
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

    main(model, val_dataloader)
