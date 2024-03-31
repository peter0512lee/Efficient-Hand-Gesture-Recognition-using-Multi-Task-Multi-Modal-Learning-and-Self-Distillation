import cv2
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


dataset_path = "C:/Users/CGVMISLAB/datasets/NvGesture/"

# MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Large"

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


def generate_depth(IMAGE_FILES, root_path):
    for idx, file in enumerate(IMAGE_FILES):
        save_path = os.path.join(
            root_path, 'sk_depth_est_all').replace('\\', '/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(
            save_path, file.split('/')[-1]).replace('\\', '/')
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
    file_with_split_list = [
        "nvgesture_train_correct_cvpr2016_v2.lst",
        "nvgesture_test_correct_cvpr2016_v2.lst"
    ]
    for file_with_split in file_with_split_list:
        print("Processing: ", file_with_split)
        file_with_split = os.path.join(dataset_path, file_with_split)
        with open(file_with_split, 'rb') as f:
            for line in f:  # subject
                params = line.decode().split(' ')
                path = params[0].split(':')[1].split('/')
                path = os.path.join(
                    dataset_path, path[1], path[2], path[3]).replace('\\', '/')
                rgb_path = os.path.join(
                    path, 'sk_color_all').replace('\\', '/')
                print(rgb_path)
                rgb = []
                start = int(params[2].split(':')[2])
                end = int(params[2].split(':')[3])
                for i in range(start, end+1):
                    rgb.append(os.path.join(rgb_path, '%05d.jpg' %
                                            i).replace('\\', '/'))
                generate_depth(rgb, path)


if __name__ == '__main__':
    main()
