
import torch

import torch.nn.functional as F
from .BasicBlock import BFCM
from torch import nn

from .GCN import gen_A,gen_adj,GraphConvolution
import pickle
from torch.nn import Parameter
from torchvision.models import resnet34,ResNet34_Weights
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU()
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)

        return out

def mask_random_patches(B, C, H, W,patch_size=16, ratio=1, mask_value=0):
    """
    随机掩码图像中的patch区域，并控制覆盖率

    参数：
    - image (torch.Tensor): 输入图像tensor，形状为 (C, H, W)
    - patch_size (int): patch的大小（假设是正方形）
    - ratio (float): 覆盖率 (0 到 1 之间)
    - mask_value (int or float): 用于掩码的值

    返回：
    - masked_image (torch.Tensor): 掩码后的图像
    """
    # def __init__(self, ):
    #     super(mask_random_patches,self).__init__()
    noise = torch.normal(0, 1, size=(3, 16, 16))

    mask_value = noise
    # 确定能划分的patch数目
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # 生成所有可能的patch位置
    patch_positions = [(i, j) for i in range(num_patches_h) for j in range(num_patches_w)]

    # 计算需要掩码的patch数量
    num_patches_to_mask = int(len(patch_positions) * ratio)

    # 随机选择patch位置进行掩码
    patches_to_mask = np.random.choice(len(patch_positions), num_patches_to_mask, replace=False)

    masked_image = torch.ones((B,C, H, W))

    for idx in patches_to_mask:
        i, j = patch_positions[idx]
        # 计算patch的顶点坐标
        start_h = i * patch_size
        start_w = j * patch_size
        end_h = start_h + patch_size
        end_w = start_w + patch_size

        # 覆盖掩码
        masked_image[:,:, start_h:end_h, start_w:end_w] = mask_value

    return masked_image


class Up(nn.Module):  # 将x1上采样，然后调整为x2的大小
    """Upscaling then double conv"""

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 将传入数据上采样，

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  # 填充为x2相同的大小
        return x1

import json
class SSL_MISS_Net(nn.Module):

    def __init__(self, block=BasicBlock, adj_path=None, num_classes=1000, in_channel=300, t=0.1,
                 p=0.25, pretrain = False):
        self.inplanes = 64
        super(SSL_MISS_Net, self).__init__()
        # ############################  Flair  ####################
        self.limage = nn.Parameter(torch.randn((1, 2, 3, 256, 256)))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #
        self.resnet_flair = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # ############################ T1C ########################
        self.resnet_t1c = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        self.gc1 = GraphConvolution(in_channel, 384)
        self.gc2 = GraphConvolution(384, 512)


        self.relu = nn.LeakyReLU(0.2)
        # Load data for adjacency matrix
        with open(adj_path,'rb') as fp:
            adj_data = pickle.load(fp)


        adj = gen_A(num_classes, t, p, adj_data)
        self.A = Parameter(torch.from_numpy(adj).float(), requires_grad=False)

        self.fusion4 = BFCM(embedding_dim=512, volumn_size=16)
        nb_filter = [64,64,128,256,512]
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion)
        self.conv2_2 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion)
        self.conv1_3 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion)
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])
        self.up1conv = nn.Conv2d(nb_filter[0], 6, (1, 1))
        self.Up = Up()

        self.pretrain = pretrain
        self.mask = nn.Parameter(mask_random_patches(2, 3, 256, 256))
        self.encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.encoder.conv1 = nn.Conv2d(6, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)


    def forward_features_T1c(self, y: torch.Tensor) -> torch.Tensor:
        y_all= []
        y = self.resnet_t1c.conv1(y)
        y = self.resnet_t1c.bn1(y)
        y0 = self.resnet_t1c.relu(y)
        # y0 = self.resnet_t1c.maxpool(y)
        y_all.append(y0)
        y1 = self.resnet_t1c.layer1(y0)
        y_all.append(y1)
        y2 = self.resnet_t1c.layer2(y1)
        y_all.append(y2)
        y3 = self.resnet_t1c.layer3(y2)
        y_all.append(y3)
        y4 = self.resnet_t1c.layer4(y3)
        y_all.append(y4)
        return y_all

    ############################  Flair  ####################
    def forward_features_Flair(self, x: torch.Tensor) -> torch.Tensor:
        x_all = []
        x = self.resnet_flair.conv1(x)  # (130,64,112,112)
        x = self.resnet_flair.bn1(x)
        x0 = self.resnet_flair.relu(x)

        x_all.append(x0)
        x1 = self.resnet_flair.layer1(x0)
        x_all.append(x1)
        x2 = self.resnet_flair.layer2(x1)
        x_all.append(x2)
        x3 = self.resnet_flair.layer3(x2)
        x_all.append(x3)
        x4 = self.resnet_flair.layer4(x3)
        x_all.append(x4)
        return x_all

    def Decoder(self, e: torch.Tensor) -> torch.Tensor:

        e = self.encoder.conv1(e)
        e = self.encoder.bn1(e)
        e0 = self.encoder.relu(e)

        e1 = self.encoder.layer1(e0)

        e2 = self.encoder.layer2(e1)

        e3 = self.encoder.layer3(e2)

        e4 = self.encoder.layer4(e3)

        x0_0 = e0
        x1_0 =e1
        x2_0 = e2
        x3_0 = e3
        x4_0 = e4
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0, x3_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.Up(x3_1, x2_0)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.Up(x2_2, x1_0)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.Up(x1_3, x0_0)], 1))
        x_final = self.up(x0_4)
        x_final = self.up1conv(x_final)
        return x_final

    def forward(self,t1c_input,flair_input,m_d,inp):
        image_t1ce_ = t1c_input
        image_flair = flair_input
        content_share_c4 = []
        B = image_flair.shape[0]
        if self.pretrain:
            if m_d == 0:
                image_flair = image_flair * self.mask[:B, :, :, :]
            elif m_d == 1:
                image_t1ce_ = image_t1ce_ * self.mask[:B, :, :, :]
        if self.pretrain:
            images_re = torch.cat([image_flair,image_t1ce_],dim=1)
            re_images = self.Decoder(images_re)
        else:
            for id in range(len(m_d)):
                if m_d[id]==0 or m_d[id]==1:
                    if m_d[id] == 0:
                        image_flair_re = image_flair[id,:,:,:]+self.mask[1, :, :, :]
                        image_t1ce__re = image_t1ce_[id,:,:,:]
                    elif m_d[id] == 1:
                        image_t1ce__re = image_t1ce_[id,:,:,:]+self.mask[1, :, :, :]
                        image_flair_re = image_flair[id, :, :, :]
                    images_re = torch.cat([image_flair_re, image_t1ce__re], dim=0).unsqueeze(0)
                    re_images = self.Decoder(images_re)
                    if m_d[id]==0:
                        image_flair[id,:,:,:] = re_images[:,:3,:,:].squeeze(0)
                    if m_d[id]==1:
                        image_t1ce_[id,:,:,:] = re_images[:,3:,:,:].squeeze(0)

            content_flair = self.forward_features_Flair(image_flair)
            content_t1ce_ = self.forward_features_T1c(image_t1ce_)
            content_share_c4.append(content_t1ce_[4])
            content_share_c4.append(content_flair[4])
            content_share_f4,atten_map = self.fusion4(content_share_c4)

            z_fea = self.avgpool(content_share_f4)
            feature = z_fea.view(z_fea.size(0), -1)

            inp = inp[0].squeeze()
            adj = gen_adj(self.A).detach()
            x = self.gc1(inp, adj)
            x = self.relu(x)
            x = self.gc2(x, adj)

            x = x.transpose(0, 1)

            x = torch.matmul(feature, x)

        # return self.sigm(x)
        if self.pretrain:

            return re_images
        else:
            return x


