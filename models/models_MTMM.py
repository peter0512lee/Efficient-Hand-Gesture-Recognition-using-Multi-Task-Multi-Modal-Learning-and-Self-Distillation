'''
MTMM
'''

from torch import nn
from models.basic_ops import ConsensusModule
from models.spatial_transforms import *
# from basic_ops import ConsensusModule
# from spatial_transforms import *
from torch.nn.init import normal_, constant_
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision
import torch
import pdb


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, img_feature_dim=112,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False,
                 modal='rgb_depth'):
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
        self.modal = modal

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
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        return_nodes = {
            'maxpool': 'maxpool',
            'layer4': 'layer4',
            'avgpool': 'avgpool',
            'fc': 'fc',
        }
        self.feature_extractor = create_feature_extractor(
            self.base_model, return_nodes=return_nodes)

        self.consensus = ConsensusModule(consensus_type)

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

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(
                True if self.pretrain == 'imagenet' else False)
            if self.is_shift:
                print('Adding action...')
                from models.action import make_temporal_shift
                # from action import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modal.find('depth') != -1:
                self.global_decoder = nn.Sequential(
                    nn.Conv2d(2048, 256, kernel_size=3,
                              stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='nearest'),

                    nn.Conv2d(256, 64, kernel_size=3,
                              stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='nearest'),

                    nn.Conv2d(64, 32, kernel_size=3, stride=1,
                              padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='nearest'),

                    nn.Conv2d(32, 32, kernel_size=3, stride=1,
                              padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                    nn.Sigmoid()
                )
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

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
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose2d):
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

    def forward(self, input):
        # [N, T, C, H, W]
        sample_len = 3 * self.new_length
        x = input.view((-1, sample_len) + input.size()[-2:])
        # [NT, C, H, W]

        # Feature Extraction
        feature_dict = self.feature_extractor(x)
        global_in = feature_dict['layer4']      # [N*S, 2048, 7, 7]

        # Class
        x = feature_dict['fc']
        x = self.new_fc(x)
        x = x.view((-1, self.num_segments) + x.size()[1:])
        output = self.consensus(x)
        output = output.squeeze(1)

        # Depth
        if self.modal.find('depth') != -1:
            global_depth_out = self.global_decoder(global_in)

        if self.modal == 'rgb':
            return output
        elif self.modal == 'rgb_depth':
            return output, global_depth_out


if __name__ == '__main__':
    net = TSN(83, 8, 'RGB', is_shift=True, base_model='resnet50', shift_div=8,
              img_feature_dim=224, consensus_type='avg', fc_lr5=True, modal='rgb')
    # print(net)
    print(net.base_model.layer4[-1])
