import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision
import random
import matplotlib.pyplot as plt
from util.diffusion_utils import *
import kornia.augmentation as K


class AugmentPipe(torch.nn.Module):
    def __init__(self, init_p=0,
        xflip=0, rotate90=0, xint=0, xint_max=0.125,
        scale=0, rotate=0, aniso=0, xfrac=0, scale_std=0.2, rotate_max=10, aniso_std=0.2, xfrac_std=0.125,
        brightness=0, contrast=0, gamma=0, hue=0, saturation=0, brightness_std=0.2, contrast_std=0.2, hue_max=0.4, saturation_std=0.2, gamma_std = 0.25,
        imgfilter=0, imgfilter_bands=[1,1,1,1], imgfilter_std=1, 
        noise=0, cutout=0, noise_std=0.5, cutout_size=0.5,
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

        #normalize images
        images = (images+1)*0.5

        assert isinstance(images, torch.Tensor) and images.ndim == 4
        self.device = images.device

        batch_size, num_channels, height, width = images.shape

        with_mask = masks is not None
        if with_mask: 
            assert isinstance(images, torch.Tensor) and masks.device == self.device and masks.ndim == 4
            batch_size_M, num_channels_M, height_M, width_M = masks.shape

        transforms =  nn.Sequential()
        
        if(self.xflip > 0):
            transforms.append(K.RandomHorizontalFlip(p=self.p, same_on_batch=False))

        
        if(self.rotate90 > 0):
            transforms.append(K.RandomRotation(degrees=[-90, 90],p=self.p, same_on_batch=False))

        
        if(self.xint > 0):
            #t = random.random() * self.xint_max
            transforms.append(K.RandomAffine(degrees=0, translate=(self.xint_max, self.xint_max), p=self.p, same_on_batch=False))
            
            
        if(self.rotate > 0):
            transforms.append(K.RandomAffine(degrees=self.rotate_max, p=self.p, same_on_batch=False))


        if(self.brightness > 0):
            transforms.append(K.RandomBrightness(brightness=(1.0-self.brightness_std, 1.0+self.brightness_std),p=self.p, same_on_batch=False))


        if(self.contrast > 0):
            transforms.append(K.RandomContrast(contrast=(1.0-self.contrast_std, 1.0+self.contrast_std), p=self.p, same_on_batch=False))


        if(self.hue > 0):
            transforms.append(K.RandomHue(hue=(-self.hue_max, self.hue_max), p=self.p, same_on_batch=False))


        


        if(self.saturation > 0):
            transforms.append(K.RandomSaturation(saturation=(1.0-self.saturation_std, 1.0+self.saturation_std), p=self.p, same_on_batch=False))
            

        if(self.noise > 0):
            transforms.append(K.RandomGaussianNoise(mean=0, std=self.noise_std, p=self.p, same_on_batch=False))

        
        aug = K.AugmentationSequential(transforms, random_apply=(1,len(transforms)))

        
        transformed_images = aug(images)

        #de-normalize images
        transformed_images = (transformed_images-0.5)/0.5

        #TO DO: AUGMENTATION FOR MASKS
        
        return transformed_images, masks

    def update_p(self, p):
            self.p = p