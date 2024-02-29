import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision
import random
import matplotlib.pyplot as plt
from util.diffusion_utils import *
import albumentations as A


class AugmentPipe(torch.nn.Module):
    def __init__(self, init_p=0,
        xflip=0, rotate90=0, xint=0, xint_max=0.125,
        scale=0, rotate=0, aniso=0, xfrac=0, scale_std=0.2, rotate_max=10, aniso_std=0.2, xfrac_std=0.125,
        brightness=0, contrast=0, gamma=0, hue=0, saturation=0, brightness_std=0.025, contrast_std=0.025, hue_max=45, saturation_std=5, gamma_std = 0.25,
        imgfilter=0, imgfilter_bands=[1,1,1,1], imgfilter_std=1, 
        noise=0, cutout=0, noise_std=0.05, cutout_size=0.5,
    ):
        super().__init__()
        self.p = init_p


        # Pixel blitting.
        self.xflip            = float(xflip)            # Probability multiplier for x-flip.
        self.rotate90         = float(rotate90)         # Probability multiplier for 90 degree rotations.
        self.xint             = float(xint)             # Probability multiplier for integer translation.
        self.xint_max         = float(xint_max)         # Range of integer translation, relative to image dimensions.

        # General geometric transformations.
        self.scale            = float(scale)            # Probability multiplier for isotropic scaling.
        self.rotate           = float(rotate)           # Probability multiplier for arbitrary rotation.
        self.aniso            = float(aniso)            # Probability multiplier for anisotropic scaling.
        self.xfrac            = float(xfrac)            # Probability multiplier for fractional translation.
        self.scale_std        = float(scale_std)        # Log2 standard deviation of isotropic scaling.
        self.rotate_max       = float(rotate_max)       # Range of arbitrary rotation, 1 = full circle.
        self.aniso_std        = float(aniso_std)        # Log2 standard deviation of anisotropic scaling.
        self.xfrac_std        = float(xfrac_std)        # Standard deviation of frational translation, relative to image dimensions.

        # Color transformations.
        self.brightness       = float(brightness)       # Probability multiplier for brightness.
        self.contrast         = float(contrast)         # Probability multiplier for contrast.
        self.gamma            = float(gamma)            # Probability multiplier for gamma.
        self.hue              = float(hue)              # Probability multiplier for hue rotation.
        self.saturation       = float(saturation)       # Probability multiplier for saturation.
        self.brightness_std   = float(brightness_std)   # Standard deviation of brightness.
        self.contrast_std     = float(contrast_std)     # Log2 standard deviation of contrast.
        self.hue_max          = float(hue_max)          # Range of hue rotation, 1 = full circle.
        self.saturation_std   = float(saturation_std)   # Log2 standard deviation of saturation.
        self.gamma_std        = float(gamma_std)        # standard deviation of gamma transform.

        # Image-space filtering.
        self.imgfilter        = float(imgfilter)        # Probability multiplier for image-space filtering.
        self.imgfilter_bands  = list(imgfilter_bands)   # Probability multipliers for individual frequency bands.
        self.imgfilter_std    = float(imgfilter_std)    # Log2 standard deviation of image-space filter amplification.

        # Image-space corruptions.
        self.noise            = float(noise)            # Probability multiplier for additive RGB noise.
        self.cutout           = float(cutout)           # Probability multiplier for cutout.
        self.noise_std        = float(noise_std)        # Standard deviation of additive RGB noise.
        self.cutout_size      = float(cutout_size)      # Size of the cutout rectangle, relative to image dimensions.


    def forward(self, images, masks=None, debug=False, args=None):

        random.seed()
        torch.random.seed()
    
        if debug:
            self.p = 0.5

        if self.p == 0:
           return images, masks

        assert isinstance(images, torch.Tensor) and images.ndim == 4
        self.device = images.device

        batch_size, num_channels, height, width = images.shape

        with_mask = masks is not None
        if with_mask: 
            assert isinstance(images, torch.Tensor) and masks.device == self.device and masks.ndim == 4
            batch_size_M, num_channels_M, height_M, width_M = masks.shape

        transforms = []
        
        if(self.xflip > 0):
            transforms.append(A.HorizontalFlip(p=self.p))

        
        if(self.rotate90 > 0):
            transforms.append(A.RandomRotate90(p=self.p))

        
        if(self.xint > 0):
            #t = random.random() * self.xint_max
            transforms.append(A.Affine(translate_percent=(0.0,self.xint_max),p=self.p))
            
            
        if(self.rotate > 0):
            transforms.append(A.Affine(rotate=(-self.rotate_max, self.rotate_max),p=self.p))


        if(self.brightness > 0):
            transforms.append(A.RandomBrightness(limit=self.brightness_std,p=self.p))


        if(self.contrast > 0):
            transforms.append(A.RandomContrast(limit=self.contrast_std,p=self.p))


        if(self.hue > 0):
            transforms.append(A.HueSaturationValue(hue_shift_limit=self.hue_max, sat_shift_limit=0, val_shift_limit=0,p=self.p))



        # if(self.gamma > 0):
        #     #print("rotate90: ", self.rotate90, "p: ", self.p)
        #     apply_T = random.random()
        #     #if p_scale > 0:
        #     i = torch.randint(0,2,[images.shape[0]], device=self.device) * self.gamma * self.p > apply_T 
        #     g = abs(random.gauss(1,self.gamma_std))
        #     print("gamma: ",g)
        #     images[i,:,:,:] = TF.adjust_gamma(images[i,:,:,:], g)


        # plt.subplot(10,1,9)
        # plot_grid_images(images, nrow=16, normalize=True)


        if(self.saturation > 0):
            transforms.append(A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=self.saturation_std, val_shift_limit=0,p=self.p))
            

        if(self.noise > 0):
            transforms.append(A.GaussNoise(var_limit=(0, self.noise_std), p=self.p))

        transformed_images = []
        for i in range(batch_size):
            n = random.randint(1,len(transforms))
            #print("n", n)
            #transform = A.Compose(transforms)
            transform = A.SomeOf(transforms,n,p=self.p)
            #image = images[i].detach().permute(1,2,0).cpu().numpy()
            image = images[i].permute(1,2,0)
            transformed = transform(image=image)
            #transformed_image= torch.from_numpy(transformed['image']).permute(2,0,1).unsqueeze(0).to(args.device)
            transformed_image= transformed['image'].permute(2,0,1).unsqueeze(0).to(args.device)
            #print("transofrmed image: ",transformed_image.shape)
            transformed_images.append(transformed_image)

        transformed_images = torch.cat(transformed_images, axis=0)
        #print("transofrmed images: ",transformed_images.shape)

        
        return transformed_images, masks

    def update_p(self, p):
            self.p = p