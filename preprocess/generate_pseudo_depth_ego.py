import cv2
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


label_path = 'C:/Users/CGVMISLAB/datasets/EgoGesture/labels-final-revised1'
frame_path = 'C:/Users/CGVMISLAB/datasets/EgoGesture/frames'

# MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Large"
# MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "DPT_Hybrid"
# MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
# model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

"""Move model to GPU if available"""
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

"""Load transforms to resize and normalize the image for large or small model"""
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def generate_depth(IMAGE_FILES):
    for idx, file in enumerate(IMAGE_FILES):
        save_path = file.replace(
            'Color/rgb', 'Depth_Est/depth_est')[:-4] + '.jpg'
        """Load image and apply transforms"""
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)

        """Predict and resize to original resolution"""
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()

        """Show result"""
        # plt.imshow(output, cmap="gray")
        plt.imsave(save_path, output, cmap="gray")


def main():
    sub_ids = [i for i in range(1, 51)]
    for sub_i in tqdm(sub_ids):
        frame_path_sub = os.path.join(frame_path, 'Subject{:02}'.format(sub_i))
        label_path_sub = os.path.join(label_path, 'Subject{:02}'.format(sub_i))
        assert len([name for name in os.listdir(label_path_sub) if name != '.DS_Store']) == len(
            [name for name in os.listdir(frame_path_sub)])
        for scene_i in range(1, len([name for name in os.listdir(frame_path_sub)]) + 1):
            rgb_path = os.path.join(
                frame_path_sub, 'Scene{:01}'.format(scene_i), 'Color')
            depth_est_path = os.path.join(
                frame_path_sub, 'Scene{:01}'.format(scene_i), 'Depth_Est')
            if os.path.isdir(depth_est_path) == False:
                os.mkdir(depth_est_path)
            label_path_iter = os.path.join(
                label_path_sub, 'Scene{:01}'.format(scene_i))
            assert len([name for name in os.listdir(label_path_iter) if 'csv' == name[-3::]]) == len(
                [name for name in os.listdir(rgb_path)])
            for group_i in range(1, len([name for name in os.listdir(rgb_path)]) + 1):
                rgb_path_group = os.path.join(
                    rgb_path, 'rgb{:01}'.format(group_i))
                depth_est_path_group = os.path.join(
                    depth_est_path, 'depth_est{:01}'.format(group_i))
                if os.path.isdir(depth_est_path_group) == False:
                    os.mkdir(depth_est_path_group)
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
                    rgb = []
                    for img_ind in range(int(data_note.values[data_i, 1]), int(data_note.values[data_i, 2] - 1)):
                        rgb.append(os.path.join(rgb_path_group,
                                   '{:06}.jpg'.format(img_ind)).replace('\\', '/'))
                    generate_depth(rgb)


if __name__ == '__main__':
    main()
