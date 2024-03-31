"""
SD using ACTION-Net weight
"""

from torch import nn
# from basic_ops import ConsensusModule
# from spatial_transforms import *
from models.basic_ops import ConsensusModule
from models.spatial_transforms import *
from torch.nn.init import normal_, constant_
import torchvision
import torch
import pdb


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)


def branchBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out//4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.Conv2d(middle_channel, middle_channel,
                  kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
    )


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output


class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in,
                      kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size,
                      stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out,
                      kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, img_feature_dim=112,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        # the dimension of the CNN feature to represent each frame
        self.img_feature_dim = img_feature_dim
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model,
                       self.modality,
                       self.num_segments,
                       self.new_length,
                       consensus_type,
                       self.dropout,
                       self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        self._prepare_self_distillation(num_class)

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(
            self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model,
                    self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model,
                      self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)

        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))
        self.base_model = getattr(torchvision.models, base_model)(
            True if self.pretrain == 'imagenet' else False)
        if self.is_shift:
            print('Adding action...')
            from models.action import make_temporal_shift
            # from action import make_temporal_shift
            make_temporal_shift(self.base_model, self.num_segments,
                                n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

        self.base_model.last_layer_name = 'fc'
        self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

    def _prepare_self_distillation(self, num_classes):
        expansion = 4
        self.scala1 = nn.Sequential(
            SepConv(
                channel_in=64 * expansion,
                channel_out=128 * expansion
            ),
            SepConv(
                channel_in=128 * expansion,
                channel_out=256 * expansion
            ),
            SepConv(
                channel_in=256 * expansion,
                channel_out=512 * expansion
            )
        )
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc1 = nn.Linear(512 * expansion, num_classes)

        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=128 * expansion,
                channel_out=256 * expansion,
            ),
            SepConv(
                channel_in=256 * expansion,
                channel_out=512 * expansion,
            )
        )
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc2 = nn.Linear(512 * expansion, num_classes)

        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=256 * expansion,
                channel_out=512 * expansion,
            )
        )
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc3 = nn.Linear(512 * expansion, num_classes)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_weight = []
        custom_bn = []

        conv_cnt = 0
        bn_cnt = 0
        for name, m in self.named_modules():
            if 'action' in name:
                ps = list(m.parameters())
                if 'bn' not in name:
                    custom_weight.append(ps[0])
                    if len(ps) == 2:
                        pdb.set_trace()
                else:
                    if not self._enable_pbn or bn_cnt == 1:
                        custom_bn.extend(list(m.parameters()))

            else:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    if self.fc_lr5:
                        lr5_weight.append(ps[0])
                    else:
                        normal_weight.append(ps[0])
                    if len(ps) == 2:
                        if self.fc_lr5:
                            lr10_bias.append(ps[1])
                        else:
                            normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.BatchNorm2d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm1d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm3d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif len(m._modules) == 0:
                    if len(list(m.parameters())) > 0:
                        raise ValueError(
                            "New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_weight"},
            {'params': custom_bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "custom_bn"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    def forward(self, x):
        #   [N, T, C, H, W]
        sample_len = 3 * self.new_length
        x = x.view((-1, sample_len) + x.size()[-2:])
        #   [NT, C, H, W]

        #   stem
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        #   layer1
        x = self.base_model.layer1(x)
        middle_output1 = self.scala1(x)
        middle_output1 = self.avgpool1(middle_output1)
        middle1_fea = middle_output1
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc1(middle_output1)
        # layer1 TSN
        middle_output1 = middle_output1.view(
            (-1, self.num_segments) + middle_output1.size()[1:])
        middle_output1 = self.consensus(middle_output1)
        middle_output1 = middle_output1.squeeze(1)

        #   layer2
        x = self.base_model.layer2(x)
        middle_output2 = self.scala2(x)
        middle_output2 = self.avgpool2(middle_output2)
        middle2_fea = middle_output2
        middle_output2 = torch.flatten(middle_output2, 1)
        middle_output2 = self.middle_fc2(middle_output2)
        #   layer2 TSN
        middle_output2 = middle_output2.view(
            (-1, self.num_segments) + middle_output2.size()[1:])
        middle_output2 = self.consensus(middle_output2)
        middle_output2 = middle_output2.squeeze(1)

        #   layer3
        x = self.base_model.layer3(x)
        middle_output3 = self.scala3(x)
        middle_output3 = self.avgpool3(middle_output3)
        middle3_fea = middle_output3
        middle_output3 = torch.flatten(middle_output3, 1)
        middle_output3 = self.middle_fc3(middle_output3)
        #   layer3 TSN
        middle_output3 = middle_output3.view(
            (-1, self.num_segments) + middle_output3.size()[1:])
        middle_output3 = self.consensus(middle_output3)
        middle_output3 = middle_output3.squeeze(1)

        #   final_layer
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        final_fea = x
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)   # dropout
        # x = self.new_fc(x)          # classifier
        #   Final TSN
        #   [NT, 83]
        x = x.view((-1, self.num_segments) + x.size()[1:])
        #   [N, 8, 83]
        x = self.consensus(x)
        #   [N, 1, 83]
        output = x.squeeze(1)
        #   [N, 83]

        return output, middle_output1, middle_output2, middle_output3, final_fea, middle1_fea, middle2_fea, middle3_fea


class TSN_Middle1(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, img_feature_dim=112,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False):
        super(TSN_Middle1, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        # the dimension of the CNN feature to represent each frame
        self.img_feature_dim = img_feature_dim
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model,
                       self.modality,
                       self.num_segments,
                       self.new_length,
                       consensus_type,
                       self.dropout,
                       self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        self._prepare_self_distillation(num_class)

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(
            self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model,
                    self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model,
                      self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)

        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))
        self.base_model = getattr(torchvision.models, base_model)(
            True if self.pretrain == 'imagenet' else False)
        if self.is_shift:
            print('Adding action...')
            from models.action import make_temporal_shift
            # from action import make_temporal_shift
            make_temporal_shift(self.base_model, self.num_segments,
                                n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

        self.base_model.last_layer_name = 'fc'
        self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

    def _prepare_self_distillation(self, num_classes):
        expansion = 4
        self.scala1 = nn.Sequential(
            SepConv(
                channel_in=64 * expansion,
                channel_out=128 * expansion
            ),
            SepConv(
                channel_in=128 * expansion,
                channel_out=256 * expansion
            ),
            SepConv(
                channel_in=256 * expansion,
                channel_out=512 * expansion
            )
        )
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc1 = nn.Linear(512 * expansion, num_classes)

        self.base_model.layer2 = None
        self.base_model.layer3 = None
        self.base_model.layer4 = None

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN_Middle1, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_weight = []
        custom_bn = []

        conv_cnt = 0
        bn_cnt = 0
        for name, m in self.named_modules():
            if 'action' in name:
                ps = list(m.parameters())
                if 'bn' not in name:
                    custom_weight.append(ps[0])
                    if len(ps) == 2:
                        pdb.set_trace()
                else:
                    if not self._enable_pbn or bn_cnt == 1:
                        custom_bn.extend(list(m.parameters()))

            else:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    if self.fc_lr5:
                        lr5_weight.append(ps[0])
                    else:
                        normal_weight.append(ps[0])
                    if len(ps) == 2:
                        if self.fc_lr5:
                            lr10_bias.append(ps[1])
                        else:
                            normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.BatchNorm2d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm1d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm3d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif len(m._modules) == 0:
                    if len(list(m.parameters())) > 0:
                        raise ValueError(
                            "New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_weight"},
            {'params': custom_bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "custom_bn"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    def forward(self, x):
        #   [N, T, C, H, W]
        sample_len = 3 * self.new_length
        x = x.view((-1, sample_len) + x.size()[-2:])
        #   [NT, C, H, W]

        #   stem
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        #   layer1
        x = self.base_model.layer1(x)
        middle_output1 = self.scala1(x)
        middle_output1 = self.avgpool1(middle_output1)
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc1(middle_output1)

        # layer1 TSN
        middle_output1 = middle_output1.view(
            (-1, self.num_segments) + middle_output1.size()[1:])
        middle_output1 = self.consensus(middle_output1)
        middle_output1 = middle_output1.squeeze(1)

        return middle_output1


class TSN_Middle2(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, img_feature_dim=112,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False):
        super(TSN_Middle2, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        # the dimension of the CNN feature to represent each frame
        self.img_feature_dim = img_feature_dim
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model,
                       self.modality,
                       self.num_segments,
                       self.new_length,
                       consensus_type,
                       self.dropout,
                       self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        self._prepare_self_distillation(num_class)

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(
            self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model,
                    self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model,
                      self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)

        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))
        self.base_model = getattr(torchvision.models, base_model)(
            True if self.pretrain == 'imagenet' else False)
        if self.is_shift:
            print('Adding action...')
            from models.action import make_temporal_shift
            # from action import make_temporal_shift
            make_temporal_shift(self.base_model, self.num_segments,
                                n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

        self.base_model.last_layer_name = 'fc'
        self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

    def _prepare_self_distillation(self, num_classes):
        expansion = 4
        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=128 * expansion,
                channel_out=256 * expansion,
            ),
            SepConv(
                channel_in=256 * expansion,
                channel_out=512 * expansion,
            )
        )
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc2 = nn.Linear(512 * expansion, num_classes)

        self.base_model.layer3 = None
        self.base_model.layer4 = None

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN_Middle2, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_weight = []
        custom_bn = []

        conv_cnt = 0
        bn_cnt = 0
        for name, m in self.named_modules():
            if 'action' in name:
                ps = list(m.parameters())
                if 'bn' not in name:
                    custom_weight.append(ps[0])
                    if len(ps) == 2:
                        pdb.set_trace()
                else:
                    if not self._enable_pbn or bn_cnt == 1:
                        custom_bn.extend(list(m.parameters()))

            else:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    if self.fc_lr5:
                        lr5_weight.append(ps[0])
                    else:
                        normal_weight.append(ps[0])
                    if len(ps) == 2:
                        if self.fc_lr5:
                            lr10_bias.append(ps[1])
                        else:
                            normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.BatchNorm2d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm1d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm3d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif len(m._modules) == 0:
                    if len(list(m.parameters())) > 0:
                        raise ValueError(
                            "New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_weight"},
            {'params': custom_bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "custom_bn"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    def forward(self, x):
        #   [N, T, C, H, W]
        sample_len = 3 * self.new_length
        x = x.view((-1, sample_len) + x.size()[-2:])
        #   [NT, C, H, W]

        #   stem
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        #   layer1
        x = self.base_model.layer1(x)

        #   layer2
        x = self.base_model.layer2(x)
        middle_output2 = self.scala2(x)
        middle_output2 = self.avgpool2(middle_output2)
        middle_output2 = torch.flatten(middle_output2, 1)
        middle_output2 = self.middle_fc2(middle_output2)
        #   layer2 TSN
        middle_output2 = middle_output2.view(
            (-1, self.num_segments) + middle_output2.size()[1:])
        middle_output2 = self.consensus(middle_output2)
        middle_output2 = middle_output2.squeeze(1)

        return middle_output2


class TSN_Middle3(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, img_feature_dim=112,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False):
        super(TSN_Middle3, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        # the dimension of the CNN feature to represent each frame
        self.img_feature_dim = img_feature_dim
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model,
                       self.modality,
                       self.num_segments,
                       self.new_length,
                       consensus_type,
                       self.dropout,
                       self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        self._prepare_self_distillation(num_class)

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(
            self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model,
                    self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model,
                      self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)

        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))
        self.base_model = getattr(torchvision.models, base_model)(
            True if self.pretrain == 'imagenet' else False)
        if self.is_shift:
            print('Adding action...')
            from models.action import make_temporal_shift
            # from action import make_temporal_shift
            make_temporal_shift(self.base_model, self.num_segments,
                                n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

        self.base_model.last_layer_name = 'fc'
        self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

    def _prepare_self_distillation(self, num_classes):
        expansion = 4
        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=256 * expansion,
                channel_out=512 * expansion,
            )
        )
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc3 = nn.Linear(512 * expansion, num_classes)

        self.base_model.layer4 = None

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN_Middle3, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_weight = []
        custom_bn = []

        conv_cnt = 0
        bn_cnt = 0
        for name, m in self.named_modules():
            if 'action' in name:
                ps = list(m.parameters())
                if 'bn' not in name:
                    custom_weight.append(ps[0])
                    if len(ps) == 2:
                        pdb.set_trace()
                else:
                    if not self._enable_pbn or bn_cnt == 1:
                        custom_bn.extend(list(m.parameters()))

            else:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    if self.fc_lr5:
                        lr5_weight.append(ps[0])
                    else:
                        normal_weight.append(ps[0])
                    if len(ps) == 2:
                        if self.fc_lr5:
                            lr10_bias.append(ps[1])
                        else:
                            normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.BatchNorm2d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm1d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm3d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif len(m._modules) == 0:
                    if len(list(m.parameters())) > 0:
                        raise ValueError(
                            "New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_weight"},
            {'params': custom_bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "custom_bn"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    def forward(self, x):
        #   [N, T, C, H, W]
        sample_len = 3 * self.new_length
        x = x.view((-1, sample_len) + x.size()[-2:])
        #   [NT, C, H, W]

        #   stem
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        #   layer1
        x = self.base_model.layer1(x)
        #   layer2
        x = self.base_model.layer2(x)
        #   layer3
        x = self.base_model.layer3(x)
        middle_output3 = self.scala3(x)
        middle_output3 = self.avgpool3(middle_output3)
        middle_output3 = torch.flatten(middle_output3, 1)
        middle_output3 = self.middle_fc3(middle_output3)

        #   layer3 TSN
        middle_output3 = middle_output3.view(
            (-1, self.num_segments) + middle_output3.size()[1:])
        middle_output3 = self.consensus(middle_output3)
        middle_output3 = middle_output3.squeeze(1)

        return middle_output3


if __name__ == '__main__':
    net = TSN(83, 8, 'RGB', is_shift=True, base_model='resnet50', shift_div=8,
              img_feature_dim=224, consensus_type='avg', fc_lr5=True)
    print(net)

    input = torch.randn(1, 8, 3, 224, 224)
    output, middle_output1, middle_output2, middle_output3, \
        final_fea, middle1_fea, middle2_fea, middle3_fea = net(input)
