import os
import time
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from tqdm import tqdm, trange
import wandb
import datetime
import logging
import torchshow as ts
from copy import deepcopy

from models.spatial_transforms import *
from models.temporal_transforms import *
from data import dataset_EgoGesture, dataset_NvGesture
import utils as utils
from models import models as TSN_model
import argparse

import av
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from transformers import TrainingArguments, Trainer
from transformers.utils import send_example_telemetry
from huggingface_hub import hf_hub_download
import evaluate

import warnings
warnings.filterwarnings("ignore")

os.environ['WANDB_MODE'] = 'disabled'

dataset = "EgoGesture"
annot_path = './data/{}_annotation'.format(dataset)
device = 'cuda:0'


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_metrics(eval_pred):
    # the compute_metrics function takes a Named Tuple as input:
    # predictions, which are the logits of the model as Numpy arrays,
    # and label_ids, which are the ground-truth labels as Numpy arrays.
    metric = evaluate.load("accuracy")
    """Computes accuracy on a batch of predictions."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example[0] for example in examples]
    )
    labels = torch.tensor([example[2] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    best_acc = 0.
    best_ema_acc = 0.

    # Seed everything
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(0)

    now = datetime.datetime.now()
    tinfo = "%d-%d-%d-%d-%d-%d" % (now.year, now.month,
                                   now.day, now.hour, now.minute, now.second)
    if dataset == 'EgoGesture':
        exp_path = "runs/EgoGesture/Baseline/"
    elif dataset == 'NvGesture':
        exp_path = "runs/NvGesture/Baseline/"

    input_mean = [.485, .456, .406]
    input_std = [.229, .224, .225]
    normalize = GroupNormalize(input_mean, input_std)
    scales = [1, .875, .75, .66]

    base_model = 'resnet50'
    clip_len = 16
    if dataset == 'EgoGesture':
        trans_train = torchvision.transforms.Compose([
            GroupScale([224, 224]),
            GroupMultiScaleCrop([224, 224], scales),
            Stack(roll=(base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(base_model not in [
                                'BNInception', 'InceptionV3'])),
            normalize,
        ])

        temporal_transform_train = torchvision.transforms.Compose([
            TemporalUniformCrop_train(clip_len)
        ])

        trans_test = torchvision.transforms.Compose([
            GroupScale([224, 224]),
            Stack(roll=(base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(base_model not in [
                                'BNInception', 'InceptionV3'])),
            normalize,
        ])

        temporal_transform_test = torchvision.transforms.Compose([
            TemporalUniformCrop_val(clip_len)
        ])
    elif dataset == 'NvGesture':
        trans_train = torchvision.transforms.Compose([
            GroupScale(256),
            GroupMultiScaleCrop(224, scales),
            Stack(roll=(base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(base_model not in [
                                'BNInception', 'InceptionV3'])),
            normalize,
        ])
        temporal_transform_train = torchvision.transforms.Compose([
            TemporalUniformCrop_train(clip_len)
        ])
        trans_test = torchvision.transforms.Compose([
            GroupScale(256),
            GroupCenterCrop(224),
            Stack(roll=(base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(base_model not in [
                                'BNInception', 'InceptionV3'])),
            normalize,
        ])
        temporal_transform_test = torchvision.transforms.Compose([
            TemporalUniformCrop_val(clip_len)
        ])

    cudnn.benchmark = True
    print("Loading dataset")
    if dataset == 'EgoGesture':
        train_dataset = dataset_EgoGesture.dataset_video_original(
            annot_path,
            'train_plus_val',
            spatial_transform=trans_train,
            temporal_transform=temporal_transform_train,
        )
        val_dataset = dataset_EgoGesture.dataset_video_original(
            annot_path,
            'test',
            spatial_transform=trans_test,
            temporal_transform=temporal_transform_test,
        )
    elif dataset == 'NvGesture':
        train_dataset = dataset_NvGesture.dataset_video_original(
            annot_path,
            'train',
            spatial_transform=trans_train,
            temporal_transform=temporal_transform_train
        )
        val_dataset = dataset_NvGesture.dataset_video_original(
            annot_path,
            'test',
            spatial_transform=trans_test,
            temporal_transform=temporal_transform_test,
        )

    with open('./data/EgoGesture_annotation/classIndAll.txt') as f:
        classIndAll = f.readlines()
        classIndAll = [i.strip().split(' ') for i in classIndAll]
    id2label = {int(i[0])-1: i[1] for i in classIndAll}
    label2id = {i[1]: int(i[0])-1 for i in classIndAll}
    print(f"Unique classes: {list(label2id.keys())}.")

    # pre-trained model from which to fine-tune
    # model_ckpt = "MCG-NJU/videomae-base"
    model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
    batch_size = 4  # batch size for training and evaluation
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        ignore_mismatched_sizes=True,
    )
    print(model)

    model_name = model_ckpt.split("/")[-1]
    # new_model_name = f"{model_name}-finetuned-egogesture"
    new_model_name = f"videomae-base-finetuned-egogesture"
    num_epochs = 50

    args = TrainingArguments(
        new_model_name,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
        max_steps=(19184 // batch_size) * num_epochs,
    )
    print(args)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train()
    print(train_results)

    # trainer.evaluate(val_dataset)

    # trainer.save_model()
    test_results = trainer.evaluate(val_dataset)
    print(test_results)

    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    trainer.save_state()

    # trainer.push_to_hub()


if __name__ == '__main__':
    main()
