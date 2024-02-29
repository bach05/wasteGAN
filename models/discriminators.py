# Discriminatiors Code
import torch
import torch.nn as nn 
from util.unet import DoubleConv
from torchvision.models import efficientnet_b0



class SimpleDiscriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(SimpleDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, ndf, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf,16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        #print("[*** [D] Input size: ", input.shape)
        feat = self.main(input)
        #print("**** feat: ",feat.shape)
        return self.MLP(feat)

class SimpleDiscriminator256(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(SimpleDiscriminator256, self).__init__()
        self.ngpu = ngpu
        dropout_rate = 0.2
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf,16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        #print("[*** [D] Input size: ", input.shape)
        feat = self.main(input)
        #print("**** feat: ",feat.shape)
        return self.MLP(feat)

class TinySimpleDiscriminator256(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(TinySimpleDiscriminator256, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 4, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 4, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf, 4, 2, 1, bias=False)
            #nn.Sigmoid()
        )
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf,8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        #print("[*** [D] Input size: ", input.shape)
        feat = self.main(input)
        #print("**** feat: ",feat.shape)
        return self.MLP(feat)


class SimpleDiscriminator256Label(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(SimpleDiscriminator256Label, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc+3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        self.MLP_img = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf,16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

        self.MLP_label = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf,16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        #print("[*** [D] Input size: ", input.shape)
        feat = self.main(input)
        #print("**** feat: ",feat.shape)
        img_out = self.MLP_img(feat)
        label_out = self.MLP_label(feat)
        return img_out, label_out


class LatentDiscriminator256(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(LatentDiscriminator256, self).__init__()
        self.ngpu = ngpu
        dropout_rate = 0.2
        self.main = nn.Sequential(
            # input is 512 x 16 x 16
            nn.Conv2d(nc, int(nc/2), 4, 1, 1, bias=False),
            nn.BatchNorm2d(int(nc/2)),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(dropout_rate),
            # state size. 256 x 14 x 14
            nn.Conv2d(int(nc/2), int(nc/4), 4, 1, 1, bias=False),
            nn.BatchNorm2d(int(nc/4)),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(dropout_rate),
            # state size. 128 x 12 x 12
            nn.Conv2d(int(nc/4), int(nc/8), 4, 1, 1, bias=False),
            nn.BatchNorm2d(int(nc/8)),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(dropout_rate),
            # state size. 64 x 10 x 10
            nn.Conv2d(int(nc/8), int(nc/16), 4, 1, 1, bias=False),
            nn.BatchNorm2d(int(nc/16)),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(dropout_rate),
            # state size. 32 x 8 x 8
            nn.Conv2d(int(nc/16), int(nc/32), 4, 1, 1, bias=False),
            nn.BatchNorm2d(int(nc/32)),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(dropout_rate),
            #state size. 16 x 8 x 8
            nn.Conv2d(int(nc/32), int(nc/128), 4, 1, 1, bias=False),
            nn.BatchNorm2d(int(nc/128)),
            nn.LeakyReLU(0.2, inplace=True)
            #nn.Dropout2d(dropout_rate),
            #state size. 4 x 10 x 10
        )
            
        self.mlp_size = int(nc/128) * 10 * 10

        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.mlp_size,20),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(20,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        #print("[*** [D] Input size: ", input.shape)
        feat = self.main(input)
        #print("**** feat: ",feat.shape)
        return self.MLP(feat)



class Discriminator256(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator256, self).__init__()
        self.ngpu = ngpu
        self.dropout_rate = 0.1
        self.main = efficientnet_b0()

        print(self.main)

        #self.main.features[12][0] = nn.Conv2d(96, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #self.main.features[12][1] = nn.BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        # self.main.classifier = nn.Sequential(
        #     nn.Conv2d(512, 128, kernel_size=(3,3), stride=2, bias=False),
        #     nn.BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(128, 32, kernel_size=(2,2), stride=2, bias=False),
        #     nn.BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Flatten(),
        #     nn.Linear(in_features=288, out_features=32, bias=True),
        #     nn.Dropout(self.dropout_rate),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(in_features=32, out_features=1, bias=True),
        #     nn.Sigmoid())
        self.main.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        #print("feat: ", self.main.features(input).shape)
        return self.main(input)
