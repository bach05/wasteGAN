import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from models.deeplab.DeepLabLayers import ASPP
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNetEncoder, self).__init__()
        self.inplanes = 8
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 8, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(8),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 8, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 16, layers[1], stride = 1)
        self.layer2 = self._make_layer(block, 8, layers[2], stride = 1)
        self.layer3 = self._make_layer(block, 3, layers[3], stride = 1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        self.tanh = nn.Tanh()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.tanh(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x


class ResNetDecoder(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNetDecoder, self).__init__()
        self.inplanes = 8
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 8, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(8),
                        nn.ReLU())
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer0 = self._make_layer(block, 8, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 16, layers[1], stride = 1)
        self.layer2 = self._make_layer(block, 8, layers[2], stride = 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer3 = self._make_layer(block, 3, layers[3], stride = 1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        self.tanh = nn.Tanh()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.up1(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.up2(x)
        x = self.layer3(x)
        x = self.tanh(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x


class FeatureLayer(nn.Module):
    def __init__(self, filter_sizes_features, kernel_sizes_features, strides_features, paddings_features, input_channels=3):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Conv2d(input_channels, filter_sizes_features[0], kernel_sizes_features[0], stride=strides_features[0],
                      padding=paddings_features[0]),
            nn.PReLU(),
            nn.Dropout2d(p=0.1),
            #nn.AvgPool2d(3, stride=1),
            nn.Conv2d(filter_sizes_features[0], filter_sizes_features[1], kernel_sizes_features[1],
                      stride=strides_features[1], padding=paddings_features[1]),
            nn.PReLU(),
            nn.Dropout2d(p=0.1),
            nn.AvgPool2d(3, stride=1),
            nn.BatchNorm2d(filter_sizes_features[1]),
            nn.Conv2d(filter_sizes_features[1], filter_sizes_features[2], kernel_sizes_features[2],
                      stride=strides_features[2], padding=paddings_features[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            #nn.AvgPool2d(3, stride=1),
            nn.Conv2d(filter_sizes_features[2], filter_sizes_features[3], kernel_sizes_features[3], stride=strides_features[3], padding=paddings_features[3]),
            nn.PReLU()
            #nn.Conv2d(filter_sizes_features[3], 1, kernel_size=2)
        )
        

        for m in self.fusion:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        return self.fusion(x)


class DeepLabModule(nn.Module):
    def __init__(self, filter_sizes_features, kernel_sizes_features, strides_features, paddings_features, input_channels=3, vis=False, aspp_dilate=[12, 24, 36], num_classes = 10,  pyr_sizes = [24, 18, 8]):
        super(DeepLabModule, self).__init__()

        high_leve_channels=filter_sizes_features[3]

        self.features_extractor = FeatureLayer(filter_sizes_features, kernel_sizes_features, strides_features, paddings_features, input_channels)

        ll_ch, hl_ch, out_ch = pyr_sizes

        self.project = nn.Sequential(
            nn.Conv2d(input_channels, ll_ch, 1, bias=False),
            #nn.BatchNorm2d(ll_ch),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(high_leve_channels, aspp_dilate, output_ch = hl_ch)
        self.classifier = nn.Sequential(
            nn.Conv2d(ll_ch+hl_ch, out_ch, 3, padding=1, bias=False),
            #nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, num_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        feature_low = features.clone()
        #print(feature_low.shape)
        features_high = self.features_extractor(features)
        #print(features_high.shape)
        low_level_feature = self.project(feature_low)
        #print(low_level_feature.shape)
        output_feature = self.aspp(features_high)
        #print(output_feature.shape)
        layers = []
        for l in range(output_feature.shape[1]):
            of = F.interpolate(output_feature[:,l,:,:].unsqueeze(1), size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
            layers.append(of)

        output_feature = torch.cat(layers,1).cuda()

        #output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       #align_corners=False)
        #print(output_feature.shape)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 16, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(16),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 16, layers[0], stride = 2)
        self.layer1 = self._make_layer(block, 32, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 64, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 128, layers[3], stride = 2)
        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x

class basicSemSegModel(nn.Module):
    def __init__(self, device=None, num_classes=5):
        super(basicSemSegModel, self).__init__()
        self.backbone = ResNet(ResidualBlock, [3, 4, 6, 3])

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        #64x16x16
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        #32x64x64
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        #32x256x256

        filter_sizes_features = [32, 48, 64, 128]
        kernel_sizes_features = [3, 3, 3, 3]
        strides_features = [1, 1, 1, 1]
        paddings_features = [2, 2, 2, 2]

        self.head = DeepLabModule(filter_sizes_features, kernel_sizes_features, strides_features, paddings_features, input_channels = 32, num_classes=num_classes)

    def forward(self, input):
        back = self.backbone(input)
        #print("******************")
        #print("backbone size: ", back.shape)
        #print("back: ",np.unique(back.detach().cpu().numpy()))

        x = self.up1(back)
        x = self.up2(x)
        x = self.up3(x)
        #print("up3 feat size: ", x.shape)
        #print("up3: ",np.unique(x.detach().cpu().numpy()))
        # plt.figure("Up")
        
        # img = vutils.make_grid(x[0].unsqueeze(1), padding=2, nrow=5, normalize=True)
        # img = img.permute(1,2,0).detach().cpu().numpy()
        # plt.imshow(img)

        x = self.head(x)
        #print("head size: ", x.shape)
        #print("head: ",np.unique(x.detach().cpu().numpy()))
        #print("******************")

        # plt.figure("Head")
        # img = vutils.make_grid(x[0].unsqueeze(1), padding=2, nrow=5, normalize=True)
        # img = img.permute(1,2,0).detach().cpu().numpy()
        # plt.imshow(img)

        # plt.figure("ARGMAX")
        # img = torch.argmax(x[0].clone(), dim=0)
        # img = img.detach().cpu().numpy()
        # plt.imshow(img)
        
        # plt.show()




        return x, back