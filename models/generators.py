# Generator Code
import torch
import torch.nn as nn
from util.unet import Up, Down, DoubleConv, OutConv
import numpy as np

class GeneratorBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, pad=0):
        super(GeneratorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d( in_ch, out_ch, kernel_size, stride, pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, input):
        return self.block(input)

class SkipGenerator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(SkipGenerator, self).__init__()

        self.block1 = GeneratorBlock(nz, ngf * 2, 4, 1, 0)
        self.block2 = GeneratorBlock(ngf * 2, ngf*4, 4, 2, 1)
        self.block3 = GeneratorBlock(ngf * 4, ngf*8, 4, 2, 1)
        self.block4 = GeneratorBlock(ngf * 8, ngf*16, 4, 2, 1)
        self.block5 = GeneratorBlock(ngf * 16, ngf*8, 4, 2, 1)
        self.block6 = GeneratorBlock(ngf * 8, ngf*4, 4, 2, 1)
        #self.block7 = GeneratorBlock(ngf * 4, ngf*2, 4, 2, 1)
        self.endBlock = nn.Sequential(
            nn.ConvTranspose2d( ngf*4, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def Up(self, input, factor):
         up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
         return up(input)

    def forward(self, input):
        x1 = self.block1(input)
        #print("***X1: ", x1.shape)
        x2 = self.block2(x1)
        #print("***X2: ", x2.shape)
        x3 = self.block3(x2) 
        #print("***X3: ", x3.shape)
        x4 = self.block4(x3)
        #print("***X4: ", x4.shape)
        x5 = self.block5(x4) 
        #print("***X5: ", x5.shape)
        x6 = self.block6(x5) 
        #print("***X6: ", x6.shape) 
        out = self.endBlock(x6) 
        #print("***OUT: ", out.shape)
        return out
        

class SimpleGenerator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(SimpleGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # input is Z, going into a convolution
            nn.ConvTranspose2d( ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class UnetGenerator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, bilinear=True, device=None):
        super(UnetGenerator, self).__init__()
        self.ngpu = ngpu
        self.mean = 0
        self.std = 0.3
        self.device = device
        
        filters_size  = [3, ngf, ngf*2, ngf*4, ngf*8, ngf*16, 3] 

        self.inc = DoubleConv(filters_size[0], filters_size[1])
        self.down1 = Down(filters_size[1], filters_size[2])
        self.down2 = Down(filters_size[2], filters_size[3])
        self.down3 = Down(filters_size[3], filters_size[4])
        factor = 2 if bilinear else 1
        self.down4 = Down(filters_size[4], filters_size[5] // factor)
        self.up1 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2 = Up(filters_size[4], filters_size[3] // factor, bilinear)
        self.up3 = Up(filters_size[3], filters_size[2] // factor, bilinear)
        self.up4 = Up(filters_size[2], filters_size[1], bilinear)
        self.outc = OutConv(filters_size[1], filters_size[-1])
        self.tanh = nn.Tanh()

    def forward(self, input):

        #print("input: {} - {}".format(torch.min(input), torch.max(input)))

        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #self.noise = torch.randn(x5.shape, device=self.device) * self.std + self.mean
        #x5 = 0.7*x5 + 0.3*self.noise
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        feat3 = x.clone()
        x = self.up3(x, x2)
        feat2 = x.clone()
        x = self.up4(x, x1)
        #print("x up4: {} - {}".format(torch.min(x), torch.max(x)))
        feat1 = x.clone()
        out = self.outc(x)
        #print("out: {} - {}".format(torch.min(out), torch.max(out)))
        out = self.tanh(out)

        return out, [feat3, feat2, feat1]


class MultiHeadUnetGenerator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, bilinear=True, device=None):
        super(MultiHeadUnetGenerator, self).__init__()
        self.ngpu = ngpu
        self.mean = 0
        self.std = 0.3
        self.device = device
        
        filters_size  = [3, ngf, ngf*2, ngf*4, ngf*8, ngf*16, 3] 

        self.inc = DoubleConv(filters_size[0], filters_size[1])
        self.down1 = Down(filters_size[1], filters_size[2])
        self.down2 = Down(filters_size[2], filters_size[3])
        self.down3 = Down(filters_size[3], filters_size[4])
        factor = 2 if bilinear else 1
        self.down4 = Down(filters_size[4], filters_size[5] // factor)

        self.up1_h1 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2_h1 = Up(filters_size[4], filters_size[3] // factor, bilinear)
        self.up3_h1 = Up(filters_size[3], filters_size[2] // factor, bilinear)
        self.up4_h1 = Up(filters_size[2], filters_size[1], bilinear)

        self.up1_h2 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2_h2 = Up(filters_size[4], filters_size[3] // factor, bilinear)
        self.up3_h2 = Up(filters_size[3], filters_size[2] // factor, bilinear)
        self.up4_h2 = Up(filters_size[2], filters_size[1], bilinear)

        self.up1_h3 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2_h3 = Up(filters_size[4], filters_size[3] // factor, bilinear)
        self.up3_h3 = Up(filters_size[3], filters_size[2] // factor, bilinear)
        self.up4_h3 = Up(filters_size[2], filters_size[1], bilinear)

        self.h_fusion = OutConv(filters_size[1]*3, filters_size[1])
        self.outc = OutConv(filters_size[1], filters_size[-1])
        self.tanh = nn.Tanh()



    def forward(self, input):

        #print("input: {} - {}".format(torch.min(input), torch.max(input)))

        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        noise = torch.randn(x5.shape, device=self.device) * self.std + self.mean
        x5_h1 = 0.7*x5 + 0.3*noise

        x = self.up1_h1(x5_h1, x4)
        x = self.up2_h1(x, x3)
        x = self.up3_h1(x, x2)
        xh1 = self.up4_h1(x, x1)

        noise = torch.randn(x5.shape, device=self.device) * self.std + self.mean
        x5_h2 = 0.7*x5 + 0.3*noise

        x = self.up1_h2(x5_h2, x4)
        x = self.up2_h2(x, x3)
        x = self.up3_h2(x, x2)
        xh2 = self.up4_h2(x, x1)

        noise = torch.randn(x5.shape, device=self.device) * self.std + self.mean
        x5_h3 = 0.7*x5 + 0.3*noise

        x = self.up1_h3(x5_h3, x4)
        x = self.up2_h3(x, x3)
        x = self.up3_h3(x, x2)
        xh3 = self.up4_h3(x, x1)
     
        x = torch.cat([xh1, xh2, xh3], axis=1)
        x = self.h_fusion(x)
        out = self.outc(x)
        out = self.tanh(out)

        return out, [xh1, xh2, xh3]


class MultiHeadUnetGenerator2(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, bilinear=True, device=None):
        super(MultiHeadUnetGenerator2, self).__init__()
        self.ngpu = ngpu
        self.mean = 0
        self.std = 0.3
        self.device = device
        
        filters_size  = [3, ngf, ngf*2, ngf*4, ngf*8, ngf*16, 3] 

        self.inc = DoubleConv(filters_size[0], filters_size[1])
        self.down1 = Down(filters_size[1], filters_size[2])
        self.down2 = Down(filters_size[2], filters_size[3])
        self.down3 = Down(filters_size[3], filters_size[4])
        factor = 2 if bilinear else 1
        self.down4 = Down(filters_size[4], filters_size[5] // factor)

        self.up1_h1 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2_h1 = Up(filters_size[4], filters_size[3] // factor, bilinear, dilation=2, kernel_size=3, padding=2)
        self.up3_h1 = Up(filters_size[3], filters_size[2] // factor, bilinear, dilation=2, kernel_size=3, padding=2)
        self.up4_h1 = Up(filters_size[2], filters_size[1], bilinear, dilation=2, kernel_size=3, padding=2)

        self.up1_h2 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2_h2 = Up(filters_size[4], filters_size[3] // factor, bilinear, dilation=2, kernel_size=5, padding=4)
        self.up3_h2 = Up(filters_size[3], filters_size[2] // factor, bilinear,dilation=2, kernel_size=5, padding=4)
        self.up4_h2 = Up(filters_size[2], filters_size[1], bilinear, dilation=2, kernel_size=5, padding=4)

        self.up1_h3 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2_h3 = Up(filters_size[4], filters_size[3] // factor, bilinear)
        self.up3_h3 = Up(filters_size[3], filters_size[2] // factor, bilinear)
        self.up4_h3 = Up(filters_size[2], filters_size[1], bilinear)

        self.h_fusion = OutConv(filters_size[1]*3, filters_size[1])
        self.outc = OutConv(filters_size[1], filters_size[-1])
        self.tanh = nn.Tanh()



    def forward(self, input):

        #print("input: {} - {}".format(torch.min(input), torch.max(input)))

        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        #print("x5: ",x5.shape)
        noise = torch.randn(x5.shape, device=self.device) * self.std + self.mean
        x5_h1 = 0.7*x5 + 0.3*noise

        x = self.up1_h1(x5_h1, x4)
        x = self.up2_h1(x, x3)
        x = self.up3_h1(x, x2)
        xh1 = self.up4_h1(x, x1)

        noise = torch.randn(x5.shape, device=self.device) * self.std + self.mean
        x5_h2 = 0.7*x5 + 0.3*noise

        x = self.up1_h2(x5_h2, x4)
        x = self.up2_h2(x, x3)
        x = self.up3_h2(x, x2)
        xh2 = self.up4_h2(x, x1)

        noise = torch.randn(x5.shape, device=self.device) * self.std + self.mean
        x5_h3 = 0.7*x5 + 0.3*noise

        x = self.up1_h3(x5_h3, x4)
        x = self.up2_h3(x, x3)
        x = self.up3_h3(x, x2)
        xh3 = self.up4_h3(x, x1)
     
        x = torch.cat([xh1, xh2, xh3], axis=1)
        x = self.h_fusion(x)
        out = self.outc(x)
        out = self.tanh(out)

        return out, [xh1, xh2, xh3]

class UnetGeneratorLabel(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, n_classes=6, bilinear=True):
        super(UnetGeneratorLabel, self).__init__()
        self.ngpu = ngpu
        
        filters_size  = [3, ngf, ngf*2, ngf*4, ngf*8, ngf*16, 3]

        self.inc = DoubleConv(filters_size[0], filters_size[1])
        self.down1 = Down(filters_size[1], filters_size[2])
        self.down2 = Down(filters_size[2], filters_size[3])
        self.down3 = Down(filters_size[3], filters_size[4])
        factor = 2 if bilinear else 1
        self.down4 = Down(filters_size[4], filters_size[5] // factor)
        self.up1 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2 = Up(filters_size[4], filters_size[3] // factor, bilinear)
        self.up3 = Up(filters_size[3], filters_size[2] // factor, bilinear)
        self.up4 = Up(filters_size[2], filters_size[1], bilinear)
        self.outc = OutConv(filters_size[1], filters_size[-1])
        self.tanh = nn.Tanh()

        self.up1Label = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2Label = Up(filters_size[4], filters_size[3] // factor, bilinear)
        self.up3Label = Up(filters_size[3], filters_size[2] // factor, bilinear)
        self.up4Label = Up(filters_size[2], filters_size[1], bilinear)
        self.outcLabel = OutConv(filters_size[1], n_classes)

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        feat5 = x
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feat2 = x
        x = self.up4(x, x1)
        feat1 = x
        out = self.outc(x)
        out = self.tanh(out)

        xLabel = self.up1Label(x5, x4)
        xLabel = self.up2Label(xLabel, x3)
        xLabel = self.up3Label(xLabel, x2)
        xLabel = self.up4Label(xLabel, x1)
        outLabel = self.outcLabel(xLabel)

        return out, outLabel, [feat5, feat2, feat1]


class DoubleUnetGenerator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, bilinear=True, device=None):
        super(DoubleUnetGenerator, self).__init__()
        self.ngpu = ngpu
        self.mean = 0
        self.std = 0.3
        self.device = device
        #assert mode in ['train', 'test']
        #self.mode = mode
        
        filters_size  = [3, ngf, ngf*2, ngf*4, ngf*8, ngf*16, 3] 

        self.inc_img = DoubleConv(filters_size[0], filters_size[1])
        self.down1_img = Down(filters_size[1], filters_size[2])
        self.down2_img = Down(filters_size[2], filters_size[3])
        self.down3_img = Down(filters_size[3], filters_size[4])

        self.inc_patch = DoubleConv(filters_size[0], filters_size[1])
        self.down1_patch = Down(filters_size[1], filters_size[2])
        self.down2_patch = Down(filters_size[2], filters_size[3])
        self.down3_patch = Down(filters_size[3], filters_size[4])

        factor = 2 if bilinear else 1
        self.down4_img = Down(filters_size[4], filters_size[5] // factor)
        self.down4_patch = Down(filters_size[4], filters_size[5] // factor)
        
        self.merge = OutConv(filters_size[5], filters_size[5] // factor)

        self.up1 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2 = Up(filters_size[4], filters_size[3] // factor, bilinear)
        self.up3 = Up(filters_size[3], filters_size[2] // factor, bilinear)
        self.up4 = Up(filters_size[2], filters_size[1], bilinear)
        self.outc = OutConv(filters_size[1], filters_size[-1])
        self.tanh = nn.Tanh()

    def forward(self, input_img, input_patch):

        if input_img is not None:
            x1_img = self.inc_img(input_img)
            x2_img = self.down1_img(x1_img)
            x3_img = self.down2_img(x2_img)
            x4_img = self.down3_img(x3_img)
            x5_img = self.down4_img(x4_img)

        x1 = self.inc_patch(input_patch)
        x2 = self.down1_patch(x1)
        x3 = self.down2_patch(x2)
        x4 = self.down3_patch(x3)
        x5 = self.down4_patch(x4)

        #print("x5: ", x5.shape)
        self.noise = torch.randn(x5.shape, device=self.device) * self.std + self.mean
        #x5 = 0.7*x5 + 0.3*self.noise
        if input_img is not None:
            x5 = torch.cat([x5, x5_img], dim=1)
            x5 = 0.7*self.merge(x5) + 0.3*self.noise
        else:
            x5 = 0.7*x5 + 0.3*self.noise

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        feat3 = x.clone()
        x = self.up3(x, x2)
        feat2 = x.clone()
        x = self.up4(x, x1)
        feat1 = x.clone()
        out = self.outc(x)
        #out = self.tanh(out)

        return out, [feat3, feat2, feat1]


class LatentUnetGenerator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, bilinear=True, device=None):
        super(LatentUnetGenerator, self).__init__()
        self.ngpu = ngpu
        self.mean = 0
        self.std = 0.3
        self.device = device
        
        filters_size  = [3, ngf, ngf*2, ngf*4, ngf*8, ngf*16, 3] 

        self.inc = DoubleConv(filters_size[0], filters_size[1])
        self.down1 = Down(filters_size[1], filters_size[2])
        self.down2 = Down(filters_size[2], filters_size[3])
        self.down3 = Down(filters_size[3], filters_size[4])
        factor = 2 if bilinear else 1
        self.down4 = Down(filters_size[4], filters_size[5] // factor)
        self.up1 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2 = Up(filters_size[4], filters_size[3] // factor, bilinear)
        self.up3 = Up(filters_size[3], filters_size[2] // factor, bilinear)
        self.up4 = Up(filters_size[2], filters_size[1], bilinear)
        self.outc = OutConv(filters_size[1], filters_size[-1])
        self.tanh = nn.Tanh()

    def forward(self, input):

        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #self.noise = torch.randn(x5.shape, device=self.device) * self.std + self.mean
        #x5 = 0.7*x5 + 0.3*self.noise

        latent = x5
        #print("latent:", latent.shape)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        feat3 = x.clone()
        x = self.up3(x, x2)
        feat2 = x.clone()
        x = self.up4(x, x1)
        feat1 = x.clone()
        out = self.outc(x)
        #out = self.tanh(out)

        return out, latent, [feat3, feat2, feat1]

class SelfAttention(nn.Module):
    def __init__(self, channels, size): #channel dimension and image resolution
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        #print("sa x: ", x.shape)
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        #print("sa x view1: ", x.shape)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class SSMultiHeadUnetGenerator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, bilinear=True, device=None):
        super(SSMultiHeadUnetGenerator, self).__init__()
        self.ngpu = ngpu
        self.mean = 0
        self.std = 0.3
        self.device = device
        
        filters_size  = [3, ngf, ngf*2, ngf*4, ngf*8, ngf*16, 3] 
        #self.filters_size = filters_size

        self.latent_res = 16

        self.inc = DoubleConv(filters_size[0], filters_size[1])
        self.down1 = Down(filters_size[1], filters_size[2])
        self.down2 = Down(filters_size[2], filters_size[3])
        self.down3 = Down(filters_size[3], filters_size[4])
        factor = 2 if bilinear else 1
        self.down4 = Down(filters_size[4], filters_size[5] // factor)

        self.pos_enc = torch.Tensor(self.get_positional_embeddings(self.latent_res, filters_size[5] // factor)).to(self.device)
        self.pos_enc.requires_grad = False
        #print("pos enc init", self.pos_enc.shape)


        self.sa = SelfAttention(filters_size[5] // factor, self.latent_res)

        self.up1_h1 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2_h1 = Up(filters_size[4], filters_size[3] // factor, bilinear, dilation=2, kernel_size=3, padding=2)
        self.up3_h1 = Up(filters_size[3], filters_size[2] // factor, bilinear, dilation=2, kernel_size=3, padding=2)
        self.up4_h1 = Up(filters_size[2], filters_size[1], bilinear, dilation=2, kernel_size=3, padding=2)

        self.up1_h2 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2_h2 = Up(filters_size[4], filters_size[3] // factor, bilinear, dilation=2, kernel_size=5, padding=4)
        self.up3_h2 = Up(filters_size[3], filters_size[2] // factor, bilinear,dilation=2, kernel_size=5, padding=4)
        self.up4_h2 = Up(filters_size[2], filters_size[1], bilinear, dilation=2, kernel_size=5, padding=4)

        self.up1_h3 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2_h3 = Up(filters_size[4], filters_size[3] // factor, bilinear)
        self.up3_h3 = Up(filters_size[3], filters_size[2] // factor, bilinear)
        self.up4_h3 = Up(filters_size[2], filters_size[1], bilinear)

        self.h_fusion = OutConv(filters_size[1]*3, filters_size[1])
        self.outc = OutConv(filters_size[1], filters_size[-1])
        self.tanh = nn.Tanh()

    #given the image resolution and the dimensionality of each of them, 
    #outputs a matrix where each coordinate (i,j) is the value to be
    #added to token i in dimension j
    def get_positional_embeddings(self, img_res, d):

        sequence_length = img_res**2
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        
        
        result = np.reshape(result, [img_res, img_res, d])
        #print("result: ", result.shape)
        return result

    def forward(self, input):

        #print("input: {} - {}".format(torch.min(input), torch.max(input)))
        #print("input: ", input.shape)

        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        #print("x5: ",x5.shape)
        #print("pos enc ", self.pos_enc.shape)

        pos_enc = self.pos_enc.permute(2,0,1).repeat(x5.shape[0],1,1,1)
        x5 = x5 + pos_enc
        x5 = self.sa(x5)
        
        #print("x4: ",x4.shape)
        #noise = torch.randn(x5.shape, device=self.device) * self.std + self.mean
        #x5_h1 = 0.7*x5 + 0.3*noise

        x = self.up1_h1(x5, x4)
        x = self.up2_h1(x, x3)
        x = self.up3_h1(x, x2)
        xh1 = self.up4_h1(x, x1)

        #noise = torch.randn(x5.shape, device=self.device) * self.std + self.mean
        #x5_h2 = 0.7*x5 + 0.3*noise

        x = self.up1_h2(x5, x4)
        x = self.up2_h2(x, x3)
        x = self.up3_h2(x, x2)
        xh2 = self.up4_h2(x, x1)

        #noise = torch.randn(x5.shape, device=self.device) * self.std + self.mean
        #x5_h3 = 0.7*x5 + 0.3*noise

        x = self.up1_h3(x5, x4)
        x = self.up2_h3(x, x3)
        x = self.up3_h3(x, x2)
        xh3 = self.up4_h3(x, x1)
     
        x = torch.cat([xh1, xh2, xh3], axis=1)
        x = self.h_fusion(x)
        out = self.outc(x)
        out = self.tanh(out)

        return out, [xh1, xh2, xh3]