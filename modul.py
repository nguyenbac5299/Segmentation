import torch
from torch import nn

# Feature Module
# FeatureMap Convolution
import torch.nn.functional as F


class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        outputs = self.relu(x)

        return outputs


class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()
        # block1
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 3, 64, 3, 2, 1, 1, False
        self.cbnr1 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        # block2
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 64, 3, 1, 1, 1, False
        self.cbnr2 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        # block3
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 128, 3, 1, 1, 1, False
        self.cbnr3 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # block4
        self.maxpool = nn.MaxPool2d(kernel_size, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr1(x)
        x = self.cbnr2(x)
        x = self.cbnr3(x)
        outputs = self.maxpool(x)

        return outputs


# Residual Block PSP
class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_block, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        # bottleNeckPSP
        self.add_module('block1', BottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation))

        for i in range(n_block - 1):
            self.add_module('block' + str(i + 2), BottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation))


class Conv2DMatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2DMatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batch_norm(x)

        return outputs


class BottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channel, stride, dilation):
        super(BottleNeckPSP, self).__init__()
        self.cbnr_1 = Conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                          bias=False)
        self.cbnr_2 = Conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation,
                                          dilation=dilation, bias=False)
        self.cbn_3 = Conv2DMatchNorm(mid_channels, out_channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=0)

        # skip connection
        self.cbn_residual = Conv2DMatchNorm(in_channels, out_channel, kernel_size=1, stride=stride, padding=0,
                                            dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbn_3(self.cbnr_2(self.cbnr_1(x)))
        residual = self.cbn_residual(x)

        return self.relu(conv + residual)


class BottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super(BottleNeckIdentifyPSP, self).__init__()
        self.cbnr_1 = Conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                          bias=False)
        self.cbnr_2 = Conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation,
                                          dilation=dilation, bias=False)
        self.cbn_3 = Conv2DMatchNorm(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                     bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbn_3(self.cbnr_2(self.cbnr_1(x)))
        residual = x

        return self.relu(conv + residual)


# Pyramid Pooling
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, height, width):
        super(PyramidPooling, self).__init__()

        self.height = height
        self.width = width
        out_channels = int(in_channels / len(pool_sizes))
        # pool_size=[6,3,2,1]
        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbnr_1 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                          bias=False)

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbnr_2 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                          bias=False)

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbnr_3 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                          bias=False)

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbnr_4 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                          bias=False)

    def forward(self, x):
        out_1 = self.cbnr_1(self.avpool_1(x))
        out_1 = F.interpolate(out_1, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out_2 = self.cbnr_2(self.avpool_2(x))
        out_2 = F.interpolate(out_2, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out_3 = self.cbnr_3(self.avpool_3(x))
        out_3 = F.interpolate(out_3, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out_4 = self.cbnr_4(self.avpool_4(x))
        out_4 = F.interpolate(out_4, size=(self.height, self.width), mode='bilinear', align_corners=True)

        output = torch.cat(x, out_1, out_2, out_3, out_4, dim=1)

        return output


class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super(DecodePSPFeature, self).__init__()

        # forward
        self.height = height
        self.width = width

        self.cbnr = Conv2DBatchNormRelu(in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1,
                                        dilation=1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classification = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbnr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)

        return output


class AuxilirayPSPLayers(nn.Module):
    def __init__(self, in_channels, height, width, n_classes):
        super(AuxilirayPSPLayers, self).__init__()

        # forward
        self.height = height
        self.width = width

        self.cbnr = Conv2DBatchNormRelu(in_channels, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1,
                                        bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classification = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbnr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)

        return output


class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        # parameter
        block_config = [3, 4, 6, 3]
        img_size= 475
        img_size_8=60

        # feature module
        self.feature_conv= FeatureMap_convolution()

        self.feature_res_1= ResidualBlockPSP(block_config[0], in_channels=128, mid_channels=64, out_channels=256, stride=1, dilation=1)
        self.feature_res_2= ResidualBlockPSP(block_config[1], in_channels=256, mid_channels=128, out_channels=512, stride=2, dilation=1)
        self.feature_dilated_res_1= ResidualBlockPSP(block_config[2], in_channels=512, mid_channels=256, out_channels=1024, stride=1, dilation=2)
        self.feature_dilated_res_2= ResidualBlockPSP(block_config[3], in_channels=1024, mid_channels=512, out_channels=2048, stride=1, dilation=4)

        # Pyramid Pooling Module
        self.pyramid_pooling=PyramidPooling(in_channels=2048,pool_sizes=[6,3,2,1],height=img_size_8, width=img_size_8)

        # decode module
        self.decode_feature= DecodePSPFeature(height=img_size, width=img_size, n_classes=n_classes)

        # auxloss module
        self.aux= AuxilirayPSPLayers(in_channels=1024, height=img_size, width=img_size, n_classes=n_classes)

    def forward(self,x):

        x= self.feature_conv(x)
        x= self.feature_res_1(x)
        x= self.feature_res_2(x)
        x= self.feature_dilated_res_1(x)
        output_aux= self.aux(x)
        x= self.feature_dilated_res_2(x)
        x= self.pyramid_pooling(x)
        output= self.decode_feature(x)

        return (output, output_aux)




if __name__ == "__main__":
    # x = torch.randn(1, 3, 475, 475)
    # feature_conv = FeatureMap_convolution()
    # outputs = feature_conv(x)
    # print(outputs.shape)  # torch.Size([1, 128, 119, 119])
    #
    # dummy_img = torch.rand(2, 3, 475, 475)
    net = PSPNet(21)
    print(net)
    # outputs = net(dummy_img)
    # print(outputs[0].shape)
