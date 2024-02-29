import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision
import random
import matplotlib.pyplot as plt
from util.diffusion_utils import *
import copy

class AugmentPipe(torch.nn.Module):
    def __init__(self, init_p=0,
        xflip=0, rotate90=0, xint=0, xint_max=0.125,
        scale=0, rotate=0, aniso=0, xfrac=0, scale_std=0.2, rotate_max=10, aniso_std=0.2, xfrac_std=0.125,
        brightness=0, contrast=0, gamma=0, hue=0, saturation=0, sharpness=0, brightness_std=0.2, contrast_std=0.2, hue_max=0.95, saturation_std=0.2, gamma_std = 0.25, sharpness_std=0.2,
        imgfilter=0, imgfilter_bands=[1,1,1,1], imgfilter_std=1,
        noise=0, cutout=0, noise_std=0.1, cutout_size=0.5,
    ):
        super().__init__()
        self.p = init_p

        # self.applied_transforms = {"xflip": 0, "rotate90": 0,  "xint": 0,
        #                            "rotate": 0,  "brightness": 0,  "contrast": 0,
        #                             "hue": 0,  "saturation": 0,  "noise": 0}


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
        self.sharpness        = float(sharpness)        # Probability multiplier for sharpness.
        self.hue              = float(hue)              # Probability multiplier for hue rotation.
        self.saturation       = float(saturation)       # Probability multiplier for saturation.
        self.brightness_std   = float(brightness_std)   # Standard deviation of brightness.
        self.contrast_std     = float(contrast_std)     # Log2 standard deviation of contrast.
        self.hue_max          = float(hue_max)          # Range of hue rotation, 1 = full circle.
        self.saturation_std   = float(saturation_std)   # Log2 standard deviation of saturation.
        self.gamma_std        = float(gamma_std)        # standard deviation of gamma transform.
        self.sharpness_std    = float(sharpness_std)    # standard deviation of sharpness transform.

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

        #images.requires_grad=False

        # self.applied_transforms = {"xflip": 0, "rotate90": 0,  "xint": 0,
        #                            "rotate": 0,  "brightness": 0,  "contrast": 0,
        #                             "hue": 0,  "saturation": 0,  "noise": 0}
    
        if debug:
            self.p = 1.0
            random.seed(1000)
            torch.random.manual_seed(1000)
            args.file_log.debug(f"*** RANGE*** {torch.min(images)} - {torch.max(images)}")

        if self.p == 0:
           return images, masks

        assert isinstance(images, torch.Tensor) and images.ndim == 4
        self.device = images.device

        batch_size, num_channels, height, width = images.shape

        #normalize images
        images = (images+1)*0.5

        with_mask = masks is not None
        if with_mask: 
            assert isinstance(images, torch.Tensor) and masks.device == self.device and masks.ndim == 4
            batch_size_M, num_channels_M, height_M, width_M = masks.shape
        
        if(self.xflip > 0):
            #print("hflip: ", self.xflip, "p: ", self.p)
            apply_T = random.random()
            #if p_scale > 0:
            i = torch.randint(0,2,[images.shape[0]], device=self.device)*self.xflip * self.p > apply_T
            #print("i xflip: ", i)
            
            if torch.sum(i) > 0:
                args.file_log.debug(f"[{args.idx} , {args.debug_str}] xflip: {i}")
                images[i,:,:,:] = TF.hflip(images[i,:,:,:])
                if masks is not None:
                    masks[i,:,:,:] = TF.hflip(masks[i,:,:,:])

            if torch.any(images.isnan()):
                        print("flip is NaN")

        # if debug:
        #     ax = plt.subplot(10,1,2)
        #     ax.set_title("hflip")
        #     plot_grid_images(images, nrow=16, normalize=True)
        
        if(self.rotate90 > 0):
            #print("rotate90: ", self.rotate90, "p: ", self.p)
            apply_T = random.random()
            #if p_scale > 0:
            i = torch.randint(0,2,[images.shape[0]], device=self.device) * self.rotate90 * self.p > apply_T
            #i = torch.randint(0,2,[images.shape[0]], device=self.device)
            #print(i)
            #print(f"flag rot90 {self.rotate90}, p {self.p}, apply_T {apply_T}")
            #i = i * self.rotate90 * self.p > apply_T
            #print("images rot90: ", images[i,:,:,:].shape)
            if torch.sum(i) > 0:
                args.file_log.debug(f"[{args.idx} , {args.debug_str}] rotate90: {i}")

                images[i,:,:,:] = TF.rotate(images[i,:,:,:], 90, torchvision.transforms.InterpolationMode.BILINEAR)
                if masks is not None:
                    masks[i,:,:,:] = TF.rotate(masks[i,:,:,:], 90, torchvision.transforms.InterpolationMode.NEAREST)
            
            if torch.any(images.isnan()):
                        print("rot90 is NaN")

        # if debug:
        #     ax = plt.subplot(10,1,3)
        #     ax.set_title("rotate90")
        #     plot_grid_images(images, nrow=16, normalize=True)

        
        
        if(self.xint > 0):
            #print("rotate90: ", self.rotate90, "p: ", self.p)
            apply_T = random.random()
            #if p_scale > 0:
            i = torch.randint(0,2,[images.shape[0]], device=self.device) * self.xint * self.p > apply_T 
            t = random.random() * self.xint_max
            #print("i traslate: ", i)
            if torch.sum(i) > 0:
                args.file_log.debug(f"[{args.idx} , {args.debug_str}] xint: {i}")

                images[i,:,:,:] = TF.affine(images[i,:,:,:], 0, [t,t], 1, 0, torchvision.transforms.InterpolationMode.BILINEAR)
                if masks is not None:
                    masks[i,:,:,:] = TF.affine(masks[i,:,:,:], 0, [t,t], 1, 0, torchvision.transforms.InterpolationMode.NEAREST)
                
            if torch.any(images.isnan()):
                    print("xint is NaN")

        # if debug:
        #     ax = plt.subplot(10,1,4)
        #     ax.set_title("translate")
        #     plot_grid_images(images, nrow=16, normalize=True)
            
        if(self.rotate > 0):
            #print("rotate90: ", self.rotate90, "p: ", self.p)
            apply_T = random.random()
            #if p_scale > 0:
            i = torch.randint(0,2,[images.shape[0]], device=self.device) * self.rotate * self.p > apply_T 
            r = random.random() * self.rotate_max
            #print("i rot: ", i)
            if torch.sum(i) > 0:
                args.file_log.debug(f"[{args.idx} , {args.debug_str}] rotate: {i}")

                images[i,:,:,:] = TF.affine(images[i,:,:,:], r, [0,0], 1, 0, torchvision.transforms.InterpolationMode.BILINEAR)
                if masks is not None:
                    masks[i,:,:,:] = TF.affine(masks[i,:,:,:], r, [0,0], 1, 0, torchvision.transforms.InterpolationMode.NEAREST)

            if torch.any(images.isnan()):
                        print("rotate is NaN")

        # if debug:
        #     ax = plt.subplot(10,1,5)
        #     ax.set_title("rotate")
        #     plot_grid_images(images, nrow=16, normalize=True)

        
        if(self.hue > 0):
            #print("rotate90: ", self.rotate90, "p: ", self.p)
            apply_T = random.random()
            #if p_scale > 0:
            i = torch.randint(0,2,[images.shape[0]], device=self.device) * self.hue * self.p > apply_T 
            h = (random.random() - 0.5) * self.hue_max
            img_cpy = torch.clone(images).detach()
            #print("i hue: ", i)
            if torch.sum(i) > 0:
                args.file_log.debug(f"[{args.idx} , {args.debug_str}] hue: {i}")
                images[i,:,:,:] = TF.adjust_hue(images[i,:,:,:], h)

            if torch.any(images.isnan()):
                print("hue is NaN")
                print("hue: ", h)
                print("nan images hue before: ", torch.sum(torch.isnan(img_cpy)))
                print(f"range before: {torch.min(img_cpy)} - {torch.max(img_cpy)}")
                print("waiting...")
            #print("nan images hue after: ", torch.sum(torch.isnan(images)))


        if(self.brightness > 0):
            #print("rotate90: ", self.rotate90, "p: ", self.p)
            apply_T = random.random()
            #if p_scale > 0:
            i = torch.randint(0,2,[images.shape[0]], device=self.device) * self.brightness * self.p > apply_T 
            b = abs(random.gauss(1,self.brightness_std))
            #print("i bright: ", i)
            if torch.sum(i) > 0:
                args.file_log.debug(f"[{args.idx} , {args.debug_str}] brightness: {i}")


                images[i,:,:,:] = TF.adjust_brightness(images[i,:,:,:], b)

            if torch.any(images.isnan()):
                        print("brightness is NaN")

        # if debug:
        #     ax = plt.subplot(10,1,6)
        #     ax.set_title("brightness")
        #     plot_grid_images(images, nrow=16, normalize=True)

        

        if(self.contrast > 0):
            #print("rotate90: ", self.rotate90, "p: ", self.p)
            apply_T = random.random()
            #if p_scale > 0:
            i = torch.randint(0,2,[images.shape[0]], device=self.device) * self.contrast * self.p > apply_T 
            #print("i contrast: ", i)
            c = abs(random.gauss(1,self.contrast_std))
            if torch.sum(i) > 0:
                args.file_log.debug(f"[{args.idx} , {args.debug_str}] contrast: {i}")


                images[i,:,:,:] = TF.adjust_contrast(images[i,:,:,:], c)

            if torch.any(images.isnan()):
                        print("contrast is NaN")

        # if debug:
        #     ax = plt.subplot(10,1,7)
        #     ax.set_title("contrast")
        #     plot_grid_images(images, nrow=16, normalize=True)

        

        # if debug:
        #     ax = plt.subplot(10,1,8)
        #     ax.set_title("hue")
        #     plot_grid_images(images, nrow=16, normalize=True)


        # if(self.gamma > 0):
        #     #print("rotate90: ", self.rotate90, "p: ", self.p)
        #     apply_T = random.random()
        #     #if p_scale > 0:
        #     i = torch.randint(0,2,[images.shape[0]], device=self.device) * self.gamma * self.p > apply_T 
        #     g = abs(random.gauss(1,self.gamma_std))
        #     #print("gamma: ",g)

        #     if torch.sum(i) > 0:
        #         images[i,:,:,:] = TF.adjust_gamma(images[i,:,:,:], g)
        #         args.file_log.debug(f"[{args.idx} , {args.debug_str}] gamma: {i}")


        #     if torch.any(images.isnan()):
        #         print("gamma is NaN")


        # plt.subplot(10,1,9)
        # plot_grid_images(images, nrow=16, normalize=True)


        if(self.saturation > 0):
            #print("rotate90: ", self.rotate90, "p: ", self.p)
            apply_T = random.random()
            #if p_scale > 0:
            i = torch.randint(0,2,[images.shape[0]], device=self.device) * self.saturation * self.p > apply_T 
            s = abs(random.gauss(1,self.saturation_std))
            #print("i saturation: ", i)
            #print("saturation: ", s)
            #print("images sat before: ", torch.sum(torch.isnan(images)))
            if torch.sum(i) > 0:
                args.file_log.debug(f"[{args.idx} , {args.debug_str}] saturation: {i}")


                images[i,:,:,:] = TF.adjust_saturation(images[i,:,:,:], s)

            if torch.any(images.isnan()):
                        print("sat is NaN")
            

        # if debug:
        #     ax = plt.subplot(10,1,9)
        #     ax.set_title("saturation")
        #     plot_grid_images(images, nrow=16, normalize=True)

        if(self.sharpness > 0):
            #print("rotate90: ", self.rotate90, "p: ", self.p)
            apply_T = random.random()
            #if p_scale > 0:
            i = torch.randint(0,2,[images.shape[0]], device=self.device) * self.sharpness * self.p > apply_T 
            s = abs(random.gauss(1, self.sharpness))
            #print("i saturation: ", i)
            #print("saturation: ", s)
            #print("images sat before: ", torch.sum(torch.isnan(images)))
            if torch.sum(i) > 0:
                args.file_log.debug(f"[{args.idx} , {args.debug_str}] shaperness: {i}")
                images[i,:,:,:] = TF.adjust_sharpness(images[i,:,:,:], s)

            if torch.any(images.isnan()):
                        print("sharp is NaN")


        if(self.noise > 0):
            #print("rotate90: ", self.rotate90, "p: ", self.p)
            apply_T = random.random()
            #if p_scale > 0:
            i = torch.randint(0,2,[images.shape[0]], device=self.device) * self.noise * self.p > apply_T 
            #print("i noise: ", i)
            if torch.sum(i) > 0:
                args.file_log.debug(f"[{args.idx} , {args.debug_str}] noise: {i}")

                images[i,:,:,:] = images[i,:,:,:] + torch.randn(images[i,:,:,:].shape, device=self.device) * self.noise_std

            if torch.any(images.isnan()):
                        print("noise is NaN")

        # if debug:
        #     ax = plt.subplot(10,1,10)
        #     ax.set_title("noise")
        #     plot_grid_images(images, nrow=16, normalize=True)

        #images.requires_grad=True

        #de-normalize images
        images = (images-0.5)/0.5


        return images, masks

    def update_p(self, p):
            self.p = p