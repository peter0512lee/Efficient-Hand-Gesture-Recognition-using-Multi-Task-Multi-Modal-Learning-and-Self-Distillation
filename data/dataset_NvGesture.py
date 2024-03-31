import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from copy import copy


dataset_path = "/media/mislab/DATA1/J1/NvGesture/"


def construct_annot(save_path, mode):
    annot_dict = {k: []
                  for k in ['rgb', 'depth', 'depth_est', 'skeleton', 'label']}
    if mode == 'train':
        file_with_split = "nvgesture_train_correct_cvpr2016_v2.lst"
    elif mode == 'test':
        file_with_split = "nvgesture_test_correct_cvpr2016_v2.lst"
    file_with_split = os.path.join(dataset_path, file_with_split)
    with open(file_with_split, 'rb') as f:
        dict_name = file_with_split[file_with_split.rfind('/')+1:]
        dict_name = dict_name[:dict_name.find('_')]
        for line in f:
            params = line.decode().split(' ')
            path = params[0].split(':')[1].split('/')
            path = os.path.join(
                dataset_path, path[1], path[2], path[3]).replace('\\', '/')
            rgb_path = os.path.join(path, 'sk_color_all').replace('\\', '/')
            depth_path = os.path.join(path, 'sk_depth_all').replace('\\', '/')
            depth_est_path = os.path.join(
                path, 'sk_depth_est_all').replace('\\', '/')
            skeleton_path = os.path.join(
                path, 'sk_skeleton_all').replace('\\', '/')
            rgb = []
            depth = []
            depth_est = []
            skeleton = []
            start = int(params[2].split(':')[2])
            end = int(params[2].split(':')[3])
            label = int(params[4].split(':')[1]) - 1
            for i in range(start, end+1):
                rgb.append(os.path.join(rgb_path, '%05d.jpg' %
                           i).replace('\\', '/'))
                depth.append(os.path.join(
                    depth_path, '%05d.jpg' % i).replace('\\', '/'))
                depth_est.append(os.path.join(
                    depth_est_path, '%05d.jpg' % i).replace('\\', '/'))
                skeleton.append(os.path.join(
                    skeleton_path, '%05d.npy' % i).replace('\\', '/'))
            annot_dict['rgb'].append(rgb)
            annot_dict['depth'].append(depth)
            annot_dict['depth_est'].append(depth_est)
            annot_dict['skeleton'].append(skeleton)
            annot_dict['label'].append(label)
    annot_df = pd.DataFrame(annot_dict)
    save_file = os.path.join(save_path, '{}.pkl'.format(mode))
    annot_df.to_pickle(save_file)


def construct_every_annot():
    save_path = '../data/NvGesture_annotation'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    construct_annot(save_path, 'train')
    construct_annot(save_path, 'test')


# construct_every_annot()


def load_video_original(annot_path, mode):
    # mode: train, val, test
    csv_file = os.path.join(annot_path, '{}.pkl'.format(mode))
    annot_df = pd.read_pickle(csv_file)
    rgb_samples = []
    depth_samples = []
    labels = []

    for frame_i in range(annot_df.shape[0]):
        rgb_list = annot_df['rgb'].iloc[frame_i]
        rgb_samples.append(rgb_list)
        depth_list = annot_df['depth'].iloc[frame_i]
        depth_samples.append(depth_list)
        labels.append(annot_df['label'].iloc[frame_i])

    print('{}: {} videos have been loaded'.format(mode, len(rgb_samples)))
    return rgb_samples, depth_samples, labels


def load_video(annot_path, mode):
    # mode: train, val, test
    csv_file = os.path.join(annot_path, '{}.pkl'.format(mode))
    annot_df = pd.read_pickle(csv_file)
    rgb_samples = []
    depth_samples = []
    depth_est_samples = []
    labels = []

    for frame_i in range(annot_df.shape[0]):
        rgb_list = annot_df['rgb'].iloc[frame_i]
        rgb_samples.append(rgb_list)
        depth_list = annot_df['depth'].iloc[frame_i]
        depth_samples.append(depth_list)
        depth_est_list = annot_df['depth_est'].iloc[frame_i]
        depth_est_samples.append(depth_est_list)
        labels.append(annot_df['label'].iloc[frame_i])

    print('{}: {} videos have been loaded'.format(mode, len(rgb_samples)))
    return rgb_samples, depth_samples, depth_est_samples, labels


class dataset_video_original(Dataset):
    def __init__(self, root_path, mode, spatial_transform, temporal_transform):
        self.root_path = root_path
        self.rgb_samples, self.depth_samples, self.labels = load_video_original(
            root_path, mode)
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        depth_name = self.depth_samples[idx]
        label = self.labels[idx]
        indices = [i for i in range(len(rgb_name))]
        selected_indice = self.temporal_transform(indices)
        clip_rgb_frames = []
        clip_depth_frames = []
        for i, frame_name_i in enumerate(selected_indice):
            rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
            clip_rgb_frames.append(rgb_cache)
            depth_cache = Image.open(depth_name[frame_name_i]).convert("L")
            clip_depth_frames.append(depth_cache)
        clip_rgb_frames = self.spatial_transform(clip_rgb_frames)
        clip_depth_frames = self.spatial_transform(clip_depth_frames)
        n, h, w = clip_rgb_frames.size()
        return clip_rgb_frames.view(-1, 3, h, w), clip_depth_frames.view(-1, 1, h, w), int(label)

    def __len__(self):
        return int(self.sample_num)


class dataset_video_MTMM(Dataset):
    def __init__(self, root_path, mode, spatial_transform, temporal_transform, normalize):
        self.root_path = root_path
        self.rgb_samples, self.depth_samples, self.depth_est_samples, self.labels = load_video(
            root_path, mode)
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.normalize = normalize

    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        depth_name = self.depth_samples[idx]
        depth_est_name = self.depth_est_samples[idx]
        label = self.labels[idx]
        indices = [i for i in range(len(rgb_name))]
        selected_indice = self.temporal_transform(indices)
        length = len(indices)
        last_segment = selected_indice[-1]
        next_indice = np.append(
            selected_indice[1:], last_segment+1 if last_segment+1 < length else last_segment)

        # Current segment
        clip_rgb_frames = []
        clip_depth_frames = []
        clip_depth_est_frames = []
        for i, frame_name_i in enumerate(selected_indice):
            rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
            clip_rgb_frames.append(rgb_cache)
            depth_cache = Image.open(depth_name[frame_name_i]).convert("L")
            clip_depth_frames.append(depth_cache)
            depth_est_cache = Image.open(
                depth_est_name[frame_name_i]).convert("L")
            clip_depth_est_frames.append(depth_est_cache)
        clip_rgb_frames = self.spatial_transform(clip_rgb_frames)
        clip_rgb_frames = self.normalize(clip_rgb_frames)
        clip_depth_frames = self.spatial_transform(clip_depth_frames)
        clip_depth_est_frames = self.spatial_transform(clip_depth_est_frames)
        n, h, w = clip_rgb_frames.size()
        clip_rgb_frames = clip_rgb_frames.view(-1, 3, h, w)
        clip_depth_frames = clip_depth_frames.view(-1, 1, h, w)
        clip_depth_est_frames = clip_depth_est_frames.view(-1, 1, h, w)

        # Next segment
        n_clip_depth_frames = []
        n_clip_depthest_frames = []
        for i, frame_name_i in enumerate(next_indice):
            depth_cache = Image.open(depth_name[frame_name_i]).convert("L")
            n_clip_depth_frames.append(depth_cache)
            depth_est_cache = Image.open(
                depth_est_name[frame_name_i]).convert("L")
            n_clip_depthest_frames.append(depth_est_cache)
        n_clip_depth_frames = self.spatial_transform(n_clip_depth_frames)
        n_clip_depthest_frames = self.spatial_transform(n_clip_depthest_frames)
        n, h, w = n_clip_depth_frames.size()
        n_clip_depth_frames = n_clip_depth_frames.view(-1, 1, h, w)
        n_clip_depthest_frames = n_clip_depthest_frames.view(-1, 1, h, w)

        # (N, H, W, C) -> (N, C, H, W)
        return clip_rgb_frames, clip_depth_frames, int(label), clip_depth_est_frames, n_clip_depth_frames, n_clip_depthest_frames

    def __len__(self):
        return int(self.sample_num)


class dataset_video_SD(Dataset):
    def __init__(self, root_path, mode, spatial_transform, temporal_transform):
        self.root_path = root_path
        self.rgb_samples, self.depth_samples, self.labels = load_video_original(
            root_path, mode)
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        label = self.labels[idx]
        indices = [i for i in range(len(rgb_name))]
        selected_indice = self.temporal_transform(indices)
        clip_rgb_frames = []
        for i, frame_name_i in enumerate(selected_indice):
            rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
            clip_rgb_frames.append(rgb_cache)
        clip_rgb_frames = self.spatial_transform(clip_rgb_frames)
        n, h, w = clip_rgb_frames.size()
        return clip_rgb_frames.view(-1, 3, h, w), int(label)

    def __len__(self):
        return int(self.sample_num)


class dataset_video_inference(Dataset):
    def __init__(self, root_path, mode, clip_num, spatial_transform, temporal_transform, normalize=None):
        self.root_path = root_path
        self.rgb_samples, self.depth_samples, self.depth_est_samples, self.labels = load_video(
            root_path, mode)
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.clip_num = clip_num
        self.normalize = normalize

    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        label = self.labels[idx]
        indices = [i for i in range(len(rgb_name))]
        video_rgb = []

        for win_i in range(self.clip_num):
            clip_rgb_frames = []
            selected_indice = self.temporal_transform(copy(indices))
            for frame_name_i in selected_indice:
                rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
                clip_rgb_frames.append(rgb_cache)
            clip_rgb_frames = self.spatial_transform(clip_rgb_frames)
            n, h, w = clip_rgb_frames.size()
            video_rgb.append(clip_rgb_frames.view(-1, 3, h, w))
        video_rgb = torch.stack(video_rgb)
        return video_rgb, int(label)

    def __len__(self):
        return int(self.sample_num)

    def __len__(self):
        return int(self.sample_num)


class dataset_video_case_study(Dataset):
    def __init__(self, root_path, mode, clip_num, spatial_transform, temporal_transform, normalize):
        self.root_path = root_path
        self.rgb_samples, self.depth_samples, self.depth_est_samples, self.labels = load_video(
            root_path, mode)
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.clip_num = clip_num
        self.normalize = normalize

    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        depth_name = self.depth_samples[idx]
        depth_est_name = self.depth_est_samples[idx]
        label = self.labels[idx]
        indices = [i for i in range(len(rgb_name))]
        video_rgb = []
        video_depth = []
        video_depth_est = []

        for win_i in range(self.clip_num):
            clip_rgb_frames = []
            clip_depth_frames = []
            clip_depth_est_frames = []
            selected_indice = self.temporal_transform(copy(indices))
            for frame_name_i in selected_indice:
                rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
                clip_rgb_frames.append(rgb_cache)
                depth_cache = Image.open(depth_name[frame_name_i]).convert("L")
                clip_depth_frames.append(depth_cache)
                depth_est_cache = Image.open(
                    depth_est_name[frame_name_i]).convert("L")
                clip_depth_est_frames.append(depth_est_cache)
            clip_rgb_frames = self.spatial_transform(clip_rgb_frames)
            clip_rgb_frames = self.normalize(clip_rgb_frames)
            clip_depth_frames = self.spatial_transform(clip_depth_frames)
            clip_depth_est_frames = self.spatial_transform(
                clip_depth_est_frames)
            n, h, w = clip_rgb_frames.size()
            clip_rgb_frames = clip_rgb_frames.view(-1, 3, h, w)
            clip_depth_frames = clip_depth_frames.view(-1, 1, h, w)
            clip_depth_est_frames = clip_depth_est_frames.view(-1, 1, h, w)
            video_rgb.append(clip_rgb_frames)
            video_depth.append(clip_depth_frames)
            video_depth_est.append(clip_depth_est_frames)
        video_rgb = torch.stack(video_rgb)
        video_depth = torch.stack(video_depth)
        video_depth_est = torch.stack(video_depth_est)

        return video_rgb, video_depth, int(label), rgb_name, video_depth_est

    def __len__(self):
        return int(self.sample_num)
