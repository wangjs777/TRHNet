import torch
from torch import nn
import torchvision
import math
from torch.nn import functional as F
from models.CA import CA
import torchvision.models as models

class Generator(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layer1(x[:, :, None, None])
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
class CANet(nn.Module):
    def __init__(self, code_length=12, num_classes=200, model_name='resnet50'):
        super().__init__()  # 初始化父类
        self.backbone = models.resnet50(pretrained=True)
        bits = code_length
        classlen = num_classes
        self.cls = nn.Linear(2048, num_classes)
        self.cls_loc = nn.Linear(2048, num_classes)

        if model_name == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True)
            self.backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.backbone.b1 = nn.Linear(2048, bits)
            self.backbone.b2 = nn.Linear(2048, bits)
            self.backbone.b3 = nn.Linear(2048, bits)
            self.backbone.b_cat = nn.Linear(1536, bits)

            self.backbone.fc = nn.Linear(bits, classlen)

            self.relu = nn.ReLU(inplace=True)
            self.num_ftrs = 2048
            self.feature_size = 512
            self.backbone.fc_x = nn.Linear(self.num_ftrs, classlen)
            self.W_1 = nn.Parameter(torch.Tensor(self.feature_size, self.num_ftrs))
            torch.nn.init.kaiming_uniform_(self.W_1, a=math.sqrt(5))
            self.W_2 = nn.Parameter(torch.Tensor(self.feature_size, self.num_ftrs))
            torch.nn.init.kaiming_uniform_(self.W_2, a=math.sqrt(5))
            self.W_3 = nn.Parameter(torch.Tensor(self.feature_size, self.num_ftrs))
            torch.nn.init.kaiming_uniform_(self.W_3, a=math.sqrt(5))
            self.encoder = nn.Linear(3 * self.num_ftrs, 512)
            self.encoder2 = nn.Linear(3 * self.num_ftrs, 512)
            self.netG = Generator(512)
            self.netG2 = Generator(512)
            # 为每个特征层创建对应的 CA 模块
            self.ca1 = CA(embed_dims=512)
            self.ca2 = CA(embed_dims=1024)
            self.ca3 = CA(embed_dims=2048)

            # stage 1
            self.backbone.conv_block1 = nn.Sequential(
                BasicConv(self.num_ftrs // 4, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
                BasicConv(self.feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True),
                nn.AdaptiveMaxPool2d(1),
            )

            # stage 2
            self.backbone.conv_block2 = nn.Sequential(
                BasicConv(self.num_ftrs // 2, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
                BasicConv(self.feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True),
                nn.AdaptiveMaxPool2d(1),
            )

            # stage 3
            self.backbone.conv_block3 = nn.Sequential(
                BasicConv(self.num_ftrs, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
                BasicConv(self.feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True),
                nn.AdaptiveMaxPool2d(1),
            )

            # concat features from different stages
            self.backbone.hashing_concat = nn.Sequential(
                nn.BatchNorm1d(self.feature_size * 3, affine=True),
                nn.Linear(self.feature_size * 3, self.feature_size),
                nn.BatchNorm1d(self.feature_size, affine=True),
                nn.ELU(inplace=True),
                nn.Linear(self.feature_size, bits),
            )
    def forward(self, x, compute_only_y33=False):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x1 = self.backbone.maxpool(x)
        x2 = self.backbone.layer1(x1)
        f1 = self.backbone.layer2(x2) # f1:(32, 512, 28, 28)
        f2 = self.backbone.layer3(f1) # f2:(32, 1024, 14, 14)
        f3 = self.backbone.layer4(f2) # f3:(32,2048, 7, 7)
        f1 = self.ca1(f1)
        f2 = self.ca2(f2)
        f3 = self.ca3(f3)
        if compute_only_y33 == True:
            # 仅计算 y33，避免其他操作
            f3 = self.ca3(f3)
            f33 = self.backbone.conv_block3(f3).view(-1, self.num_ftrs)
            f33_b = self.backbone.b3(f33)
            y33 = self.backbone.fc(f33_b)
            return y33
        f1 = self.ca1(f1)
        f2 = self.ca2(f2)
        f3 = self.ca3(f3)

        feats = f3

        f11 = self.backbone.conv_block1(f1).view(-1, self.num_ftrs)
        f11_b = self.backbone.b1(f11)
        deep_S_1 = F.linear(f11, self.W_1)


        f22 = self.backbone.conv_block2(f2).view(-1, self.num_ftrs)
        f22_b = self.backbone.b2(f22)
        deep_S_2 = F.linear(f22, self.W_2)

        f33 = self.backbone.conv_block3(f3).view(-1, self.num_ftrs)
        f33_b = self.backbone.b3(f33)
        y33 = self.backbone.fc(f33_b)
        deep_S_3 = F.linear(f33, self.W_3)
        all_f = torch.cat((f11, f22, f33), -1)

        deep_S = torch.cat((deep_S_1, deep_S_2, deep_S_3), -1)
        f44_b = self.backbone.hashing_concat(deep_S)
        if not self.training:
            return f44_b
        deep_S_1_dret = F.linear(deep_S_1, self.W_1.t())
        deep_S_2_dret = F.linear(deep_S_2, self.W_2.t())
        deep_S_3_dret = F.linear(deep_S_3, self.W_3.t())
        deep_S_dret = torch.cat((deep_S_1_dret, deep_S_2_dret, deep_S_3_dret), -1)
        _img = self.netG(self.encoder(all_f))
        _img2 = self.netG2(self.encoder2(deep_S_dret))
        f33 = self.cls(f33)
        f11 = self.cls_loc(f11)
        f22 = self.cls_loc(f22)
        

        # x = self.backbone.avgpool(feats)
        # x = torch.flatten(x, 1)
        # y_x = self.backbone.fc_x(x)

        return f44_b, deep_S_dret, all_f, deep_S, {'_img': _img, '_img2': _img2}, f11, f22, f33, y33, feats



# Model Setting
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x