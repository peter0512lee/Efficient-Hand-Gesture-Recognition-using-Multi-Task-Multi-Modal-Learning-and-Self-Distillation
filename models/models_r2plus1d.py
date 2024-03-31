import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

# Default Model


class R2PLUS1D_18(nn.Module):
    def __init__(self, num_classes=10):
        super(R2PLUS1D_18, self).__init__()
        self.model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes, bias=True)
        )

    def forward(self, x):
        outputs = self.model(x)
        return outputs

# MTMM Model


class R2PLUS1D_18_MTMM(nn.Module):
    def __init__(self, num_classes=10):
        super(R2PLUS1D_18_MTMM, self).__init__()
        self.model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes, bias=True)
        )
        return_nodes = {
            'stem': 'stem',
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4'
        }
        self.feature_extractor = create_feature_extractor(
            model=self.model,
            return_nodes=return_nodes
        )
        # self.local_depth_decoder = nn.Sequential(
        #     # [N, 64, 16, 56, 56] -> [N, 32, 16, 112, 112]
        #     nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(
        #         1, 2, 2), padding=(0, 1, 1), bias=False),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True),
        #     # [N, 32, 16, 112, 112] -> [N, 16, 16, 112, 112]
        #     nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm3d(16),
        #     nn.ReLU(inplace=True),
        #     # [N, 16, 16, 112, 112] -> [N, 1, 16, 112, 112]
        #     nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm3d(1),
        #     nn.ReLU(inplace=True),
        # )
        self.global_depth_decoder = nn.Sequential(
            # [N, 512, 1, 14, 14] -> [N, 256, 2, 28, 28]
            nn.ConvTranspose3d(512, 256, kernel_size=(4, 4, 4), stride=(
                2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            # # [N, 256, 2, 28, 28] -> [N, 128, 4, 56, 56]
            nn.ConvTranspose3d(256, 128, kernel_size=(4, 4, 4), stride=(
                2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            # [N, 128, 4, 56, 56] -> [N, 64, 8, 56, 56]
            nn.ConvTranspose3d(128, 64, kernel_size=(4, 1, 1), stride=(
                2, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # [N, 64, 8, 56, 56] -> [N, 32, 8, 56, 56]
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # [N, 32, 8, 56, 56] -> [N, 1, 8, 56, 56]
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feat_dict = self.feature_extractor(x)
        layer4_in = feat_dict['layer4']             # [N, 512, 1, 14, 14]
        g_depth_out = self.global_depth_decoder(
            layer4_in)   # [N, 1, 8, 56, 56]
        g_depth_out = g_depth_out.permute(0, 2, 1, 3, 4)  # [N, 8, 1, 56, 56]
        outputs = self.model(x)
        return outputs, g_depth_out


if __name__ == '__main__':
    model = R2PLUS1D_18_MTMM(83)
    # print(model)
    x = torch.randn(2, 3, 8, 224, 224)
    y, gd = model(x)
    print(y.shape)
    # print(ld.shape)
    print(gd.shape)
