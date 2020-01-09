import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation), nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(
        nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            padding=pad,
            stride=stride), nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super().__init__()

        self.conv1 = nn.Sequential(
            convbn(inplanes, planes, 3, stride, pad, dilation),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class FeatureExtraction(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.downsample = nn.ModuleList()
        in_channel = 3
        out_channel = 32
        for _ in range(k):
            self.downsample.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=5,
                    stride=2,
                    padding=2))
            in_channel = out_channel
            out_channel = 32
        self.residual_blocks = nn.ModuleList()
        for _ in range(6):
            self.residual_blocks.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=1))
        self.conv_alone = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb_img):
        output = rgb_img
        for i in range(self.k):
            output = self.downsample[i](output)
        for block in self.residual_blocks:
            output = block(output)
        return self.conv_alone(output)


class EdgeAwareRefinement(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        # self.conv2d_alone = nn.Conv2d(
        #     in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.conv2d_fea = nn.Sequential(
            convbn(in_channel, 32, kernel_size=3, stride=1, pad=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.residual_astrous_blocks = nn.ModuleList()
        astrous_list = [1, 2, 4, 8, 1, 1]
        for di in astrous_list:
            self.residual_astrous_blocks.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=di))

        self.conv2d_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, low_disparity, corresponding_rgb):
        output = torch.unsqueeze(low_disparity, dim=1)
        twice_disparity = F.interpolate(
            output,
            size=corresponding_rgb.size()[-2:],
            mode='bilinear',
            align_corners=False)
        twice_disparity *= 2
        output = self.conv2d_fea(
            torch.cat([twice_disparity, corresponding_rgb], dim=1))

        for astrous_block in self.residual_astrous_blocks:
            output = astrous_block(output)

        return nn.ReLU(inplace=True)(torch.squeeze(
            twice_disparity + self.conv2d_out(output), dim=1))


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.disp = torch.FloatTensor(
            np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


class StereoNet(nn.Module):
    def __init__(self, k, r, maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.k = k
        self.feature_extraction = FeatureExtraction(k)
        self.filter = nn.ModuleList()
        for _ in range(4):
            self.filter.append(
                nn.Sequential(
                    convbn_3d(1, 1, kernel_size=3, stride=1, pad=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)))

        self.conv3d_alone = nn.Conv3d(
            1, 1, kernel_size=3, stride=1, padding=1)

        self.edge_aware_refinements = nn.ModuleList()
        for _ in range(r):
            self.edge_aware_refinements.append(EdgeAwareRefinement(4))

    def forward(self, left, right):
        disp = (self.maxdisp + 1) // pow(2, self.k)
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        # matching
        cost = torch.FloatTensor(refimg_fea.size()[0], 1, disp,
                                 refimg_fea.size()[2],
                                 refimg_fea.size()[3]).zero_().cuda()

        for i in range(disp):
            if i > 0:
                cost[:, :, i, :, i:] = torch.cumsum(
                    refimg_fea[:, :, :, i:] * targetimg_fea[:, :, :, :-i],
                    dim=1)[:, -1:, :, :]
            else:
                cost[:, :, i, :, :] = torch.cumsum(
                    refimg_fea * targetimg_fea, dim=1)[:, -1:, :, :]
        cost = cost.contiguous()

        for f in self.filter:
            cost = f(cost)

        cost = self.conv3d_alone(cost)

        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost, dim=1)
        pred = disparityregression(disp)(pred)

        zoomed_out_refimg_fea = [left]
        for i in range(self.k - 1):
            zoomed_out_refimg_fea.append(
                F.interpolate(
                    left,
                    scale_factor=1 / pow(2, i + 1),
                    mode='bilinear',
                    align_corners=False))
        length = len(zoomed_out_refimg_fea)
        prediction_list = [pred]
        for i in range(length):
            pred = prediction_list[-1]
            prediction_list.append(self.edge_aware_refinements[i](
                pred, zoomed_out_refimg_fea[-i - 1]))

        return prediction_list
