import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from copy import copy


"""
This script creates annotation file for training, validation and testing.
Run this file first before processing anything
"""

# label files stored path
label_path = '/work/twsuakq355/J1/datasets/EgoGesture/labels-final-revised1'
# frame files stored path
frame_path = '/work/twsuakq355/J1/datasets/EgoGesture/frames'


def construct_annot(save_path, mode):
    annot_dict = {k: []
                  for k in ['rgb', 'depth', 'depth_est', 'label']}
    if mode == 'train':
        sub_ids = [3, 4, 5, 6, 8, 10, 15, 16, 17, 20, 21, 22, 23, 25, 26, 27, 30, 32, 36, 38, 39, 40, 42, 43, 44, 45,
                   46, 48, 49, 50]
    elif mode == 'val':
        sub_ids = [1, 7, 12, 13, 24, 29, 33, 34, 35, 37]
    elif mode == 'test':
        sub_ids = [2, 9, 11, 14, 18, 19, 28, 31, 41, 47]
    elif mode == 'train_plus_val':
        sub_ids = [3, 4, 5, 6, 8, 10, 15, 16, 17, 20, 21, 22, 23, 25, 26, 27, 30, 32, 36, 38, 39, 40, 42, 43, 44, 45,
                   46, 48, 49, 50, 1, 7, 12, 13, 24, 29, 33, 34, 35, 37]
    for sub_i in tqdm(sub_ids):
        frame_path_sub = os.path.join(frame_path, 'Subject{:02}'.format(sub_i))
        label_path_sub = os.path.join(label_path, 'subject{:02}'.format(sub_i))
        assert len([name for name in os.listdir(label_path_sub) if name != '.DS_Store']) == len(
            [name for name in os.listdir(frame_path_sub)])
        for scene_i in range(1, len([name for name in os.listdir(frame_path_sub)]) + 1):
            rgb_path = os.path.join(
                frame_path_sub, 'Scene{:01}'.format(scene_i), 'Color')
            depth_path = os.path.join(
                frame_path_sub, 'Scene{:01}'.format(scene_i), 'Depth')
            depth_est_path = os.path.join(
                frame_path_sub, 'Scene{:01}'.format(scene_i), 'Depth_Est')
            label_path_iter = os.path.join(
                label_path_sub, 'Scene{:01}'.format(scene_i))
            for group_i in range(1, len([name for name in os.listdir(rgb_path)]) + 1):
                rgb_path_group = os.path.join(
                    rgb_path, 'rgb{:01}'.format(group_i))
                depth_path_group = os.path.join(
                    depth_path, 'depth{:01}'.format(group_i))
                depth_est_path_group = os.path.join(
                    depth_est_path, 'depth_est{:01}'.format(group_i))
                if os.path.isfile(os.path.join(label_path_iter, 'Group{:01}.csv'.format(group_i))):
                    label_path_group = os.path.join(
                        label_path_iter, 'Group{:01}.csv'.format(group_i))
                else:
                    label_path_group = os.path.join(
                        label_path_iter, 'group{:01}.csv'.format(group_i))
                # read the annotation files in the label path
                data_note = pd.read_csv(label_path_group, names=[
                                        'class', 'start', 'end'])
                data_note = data_note[np.isnan(data_note['start']) == False]
                for data_i in range(data_note.values.shape[0]):
                    label = int(data_note.values[data_i, 0])
                    rgb = []
                    depth = []
                    depth_est = []
                    for img_ind in range(int(data_note.values[data_i, 1]), int(data_note.values[data_i, 2] - 1)):
                        rgb.append(os.path.join(rgb_path_group,
                                   '{:06}.jpg'.format(img_ind)).replace('\\', '/'))
                        depth.append(os.path.join(
                            depth_path_group, '{:06}.jpg'.format(img_ind)).replace('\\', '/'))
                        depth_est.append(os.path.join(
                            depth_est_path_group, '{:06}.jpg'.format(img_ind)).replace('\\', '/'))
                    annot_dict['rgb'].append(rgb)
                    annot_dict['depth'].append(depth)
                    annot_dict['depth_est'].append(depth_est)
                    annot_dict['label'].append(label-1)
    annot_df = pd.DataFrame(annot_dict)
    save_file = os.path.join(save_path, '{}.pkl'.format(mode))
    annot_df.to_pickle(save_file)


def construct_every_annot():
    save_path = '../data/EgoGesture_annotation'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    construct_annot(save_path, 'train')
    construct_annot(save_path, 'val')
    construct_annot(save_path, 'test')
    construct_annot(save_path, 'train_plus_val')


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
