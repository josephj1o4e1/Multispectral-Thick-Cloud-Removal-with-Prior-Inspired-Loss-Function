import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
import os
import sys
from torch import Tensor
# import sys
# sys.setrecursionlimit(10000)

CURR_DIR = os.path.abspath(os.path.dirname(__file__)) #C:/.../Indoor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        # torch.nn.init.xavier_uniform(m.weight)
        # torch.nn.init.kaiming_uniform_(m.weight) # torch.nn.init.kaiming_uniform -> torch.nn.init.kaiming_uniform_
        torch.nn.init.orthogonal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) # seems that can't use xavier nor kaiming normal
        m.bias.data.fill_(0)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    "7x7 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)

class BasicBlock_res(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_res, self).__init__()
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
class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)
class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)
class myDenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(myDenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, int(growth_rate*6/8),
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.add_module('conv2_5', nn.Conv2d(bn_size * growth_rate, int(growth_rate*1/8),
                                           kernel_size=5, stride=1, padding=2,
                                           bias=False)),
        self.add_module('conv2_7', nn.Conv2d(bn_size * growth_rate, int(growth_rate*1/8),
                                           kernel_size=7, stride=1, padding=3,
                                           bias=False)),
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)

        bottleneck_output = self.relu2(self.norm2(bottleneck_output))
        new_features = self.conv2(bottleneck_output)
        new_features_5 = self.conv2_5(bottleneck_output)
        new_features_7 = self.conv2_7(bottleneck_output)
        new_features = torch.cat([new_features, new_features_5, new_features_7], 1)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features
class myDenseblock1(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0.0, memory_efficient=False):
        super(myDenseblock1, self).__init__()
        for i in range(num_layers):
            layer = myDenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features) # bn_function will concat the current features
        return torch.cat(features, 1)

class Dense_rain_cvprw3(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw3, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0
        
        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(387,256)
        self.trans_block5=TransitionBlock(643,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(259,128)
        self.trans_block6=TransitionBlock(387,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(67,64)
        self.trans_block7=TransitionBlock(131,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(35,32)
        self.trans_block8=TransitionBlock(67,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(387,387)
        self.res42 = BasicBlock_res(387, 387)


        self.res51=BasicBlock_res(259,259)
        self.res52 = BasicBlock_res(259, 259)


        self.res61=BasicBlock_res(67,67)
        self.res62 = BasicBlock_res(67, 67)

        self.res71=BasicBlock_res(35,35)
        self.res72 = BasicBlock_res(35, 35)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        x43=F.avg_pool2d(x,16)
        # print(x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print(x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        x62 = self.res61(x62)
        x6 = self.res62(x62)
        # print(x6.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        x72 = self.res71(x72)
        x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        # x9=self.relu((self.conv_refin(x8)))
        x9=self.relu(self.batchnorm20(self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        residual = self.tanh(self.refine3(dehaze))
        # dehaze=x - residual

        return residual

class Dense_rain_cvprw3_dense161(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw3_dense161, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet161(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(1056,528)
        self.trans_block4=TransitionBlock(1056+528,256)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(643,512)
        self.trans_block5=TransitionBlock(1155,256)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(451,256)
        self.trans_block6=TransitionBlock(707,128)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(131,128)
        self.trans_block7=TransitionBlock(259,64)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(67,64)
        self.trans_block8=TransitionBlock(131,32)

        self.conv_refin=nn.Conv2d(35,36,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(36, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(36, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(36, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(36, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(36+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(36)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(1056,1056)
        self.res32 = BasicBlock_res(1056, 1056)


        self.res41=BasicBlock_res(643,643)
        self.res42 = BasicBlock_res(643, 643)


        self.res51=BasicBlock_res(451,451)
        self.res52 = BasicBlock_res(451, 451)


        self.res61=BasicBlock_res(131,131)
        self.res62 = BasicBlock_res(131,131)

        self.res71=BasicBlock_res(67,67)
        self.res72 = BasicBlock_res(67, 67)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        x43=F.avg_pool2d(x,16)
        # print(x43.size())


        x42 = torch.cat([x4,x2,x43], 1) # 256+384+3 = 643
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1) # 256+192+3 = 451
        # print(x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        x62 = self.res61(x62)
        x6 = self.res62(x62)
        # print(x6.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        x72 = self.res71(x72)
        x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        # x9=self.relu((self.conv_refin(x8)))
        x9=self.relu(self.batchnorm20(self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        residual = self.tanh(self.refine3(dehaze))
        # dehaze=x - residual

        return residual

class Dense_rain_cvprw3_357(nn.Module):
    def __init__(self, myencoder=True):
        super(Dense_rain_cvprw3_357, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        if myencoder:
            self.dense_block1 = myDenseblock1(6, 64, 4, 32)
        else:
            self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(387,256)
        self.trans_block5=TransitionBlock(643,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(259,128)
        self.trans_block6=TransitionBlock(387,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(67,64)
        self.trans_block7=TransitionBlock(131,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(35,32)
        self.trans_block8=TransitionBlock(67,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(387,387)
        self.res42 = BasicBlock_res(387, 387)


        self.res51=BasicBlock_res(259,259)
        self.res52 = BasicBlock_res(259, 259)


        self.res61=BasicBlock_res(67,67)
        self.res62 = BasicBlock_res(67, 67)

        self.res71=BasicBlock_res(35,35)
        self.res72 = BasicBlock_res(35, 35)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        x43=F.avg_pool2d(x,16)
        # print(x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print(x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        x62 = self.res61(x62)
        x6 = self.res62(x62)
        # print(x6.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        x72 = self.res71(x72)
        x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        # x9=self.relu((self.conv_refin(x8)))
        x9=self.relu(self.batchnorm20(self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        residual = self.tanh(self.refine3(dehaze))
        # dehaze=x - residual

        return residual

class Satellite4in4out_Dense_rain_cvprw3(nn.Module):
    def __init__(self):
        super(Satellite4in4out_Dense_rain_cvprw3, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        # self.conv0=haze_class.features.conv0
        self.conv0=conv7x7(4, 64, stride=2)
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0
        
        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(388,256)
        self.trans_block5=TransitionBlock(644,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(260,128)
        self.trans_block6=TransitionBlock(388,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(68,64)
        self.trans_block7=TransitionBlock(132,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(36,32)
        self.trans_block8=TransitionBlock(68,16)

        self.conv_refin=nn.Conv2d(20,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 4, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(388,388)
        self.res42 = BasicBlock_res(388, 388)


        self.res51=BasicBlock_res(260,260)
        self.res52 = BasicBlock_res(260, 260)


        self.res61=BasicBlock_res(68,68)
        self.res62 = BasicBlock_res(68, 68)

        self.res71=BasicBlock_res(36,36)
        self.res72 = BasicBlock_res(36, 36)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        x43=F.avg_pool2d(x,16)
        # print(x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print(x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        x62 = self.res61(x62)
        x6 = self.res62(x62)
        # print(x6.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        x72 = self.res71(x72)
        x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        # x9=self.relu((self.conv_refin(x8)))
        x9=self.relu(self.batchnorm20(self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        residual = self.tanh(self.refine3(dehaze))
        # dehaze=x - residual

        return residual

class Satellite4in4out_NDVI_NDWI_Dense_rain_cvprw3(nn.Module):
    def __init__(self):
        super(Satellite4in4out_NDVI_NDWI_Dense_rain_cvprw3, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        # self.conv0=haze_class.features.conv0
        self.conv0=conv7x7(4, 64, stride=2)
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0
        
        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(388,256)
        self.trans_block5=TransitionBlock(644,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(260,128)
        self.trans_block6=TransitionBlock(388,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(68,64)
        self.trans_block7=TransitionBlock(132,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(36,32)
        self.trans_block8=TransitionBlock(68,16)

        self.conv_refin=nn.Conv2d(20,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine2= nn.Conv2d(20+4, 4, kernel_size=3,stride=1,padding=1)

        self.refine3= nn.Conv2d(20+4, 24, kernel_size=3,stride=1,padding=1)
        self.refine4= nn.Conv2d(24, 4, kernel_size=7,stride=1,padding=3)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.relu2=nn.ReLU(inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(388,388)
        self.res42 = BasicBlock_res(388, 388)


        self.res51=BasicBlock_res(260,260)
        self.res52 = BasicBlock_res(260, 260)


        self.res61=BasicBlock_res(68,68)
        self.res62 = BasicBlock_res(68, 68)

        self.res71=BasicBlock_res(36,36)
        self.res72 = BasicBlock_res(36, 36)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        x43=F.avg_pool2d(x,16)
        # print(x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print(x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        x62 = self.res61(x62)
        x6 = self.res62(x62)
        # print(x6.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        x72 = self.res71(x72)
        x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        # x9=self.relu((self.conv_refin(x8)))
        x9=self.relu(self.batchnorm20(self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        dehazeImg = self.tanh(self.refine2(dehaze))

        dehazeforNDVI_NDWI = self.tanh(self.refine3(dehaze))
        dehazeforNDVI_NDWI = self.relu2(self.refine4(dehazeforNDVI_NDWI)) # need relu for not exploding(no negative values allowed when calculating NDVI/NDWI)

        return dehazeImg, dehazeforNDVI_NDWI

class Satellite4in4out_NDVI_NDWI_avoidcloud_Dense_rain_cvprw3(nn.Module):
    def __init__(self):
        super(Satellite4in4out_NDVI_NDWI_avoidcloud_Dense_rain_cvprw3, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        # self.conv0=haze_class.features.conv0
        self.conv0=conv7x7(4, 64, stride=2)
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0
        
        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(388,256)
        self.trans_block5=TransitionBlock(644,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(260,128)
        self.trans_block6=TransitionBlock(388,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(68,64)
        self.trans_block7=TransitionBlock(132,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(36,32)
        self.trans_block8=TransitionBlock(68,16)

        self.conv_refin=nn.Conv2d(20,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine2= nn.Conv2d(20+4, 4, kernel_size=3,stride=1,padding=1)

        self.refine3= nn.Conv2d(20+4, 24, kernel_size=3,stride=1,padding=1)
        self.refine4= nn.Conv2d(24, 4, kernel_size=7,stride=1,padding=3)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.relu2=nn.ReLU(inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(388,388)
        self.res42 = BasicBlock_res(388, 388)


        self.res51=BasicBlock_res(260,260)
        self.res52 = BasicBlock_res(260, 260)


        self.res61=BasicBlock_res(68,68)
        self.res62 = BasicBlock_res(68, 68)

        self.res71=BasicBlock_res(36,36)
        self.res72 = BasicBlock_res(36, 36)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        x43=F.avg_pool2d(x,16)
        # print(x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print(x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        x62 = self.res61(x62)
        x6 = self.res62(x62)
        # print(x6.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        x72 = self.res71(x72)
        x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        # x9=self.relu((self.conv_refin(x8)))
        x9=self.relu(self.batchnorm20(self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        dehazeImg = self.tanh(self.refine2(dehaze))

        dehazeforNDVI_NDWI = self.tanh(self.refine3(dehaze))
        dehazeforNDVI_NDWI = self.relu2(self.refine4(dehazeforNDVI_NDWI)) # need relu for not exploding(no negative values allowed when calculating NDVI/NDWI)

        return dehazeImg, dehazeforNDVI_NDWI

