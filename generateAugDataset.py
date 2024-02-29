import yaml
import math
from pathlib import Path
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from PIL import Image
import os, sys
from util.augment import *
import lpips

from models.styleGAN2ada import MaskDiscriminator, MaskGenerator, MaskGeneratorXL, MappingNetwork, GradientPenalty, PathLengthPenalty, BranchedMaskDiscriminator
from models.styleGAN2ada import logTanh, Sloss, Adapter, GetFeatureExtractor
from util.styleGAN import *
from munch import DefaultMunch
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from util.diffusion_utils import *
from util.customTransforms import *
from util.condDataset import PatchDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import logging 
import copy
from util.externalTools import computeAvgGrad, batchSTD, prep_dseg_output
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights 


class Configurator(object):
    """
    # Used to load a configuration YAML file, return an args object
    """
    def __init__(self, config_path, pref):

        self.config_path = config_path
        self.prefix = pref

    def getArgs(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
      
        config['device'] = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        args = DefaultMunch.fromDict(config)
        
        args.prefix = self.prefix

        if args.use_genXL: 
            args.test_id += "_genXL"
        if args.use_symLogAct: 
            args.test_id += "_symLogA{}".format(args.symLogAct_a)        
        if args.use_adaCustom: 
            args.test_id += "_adaC"
        if args.use_advCustom: 
            args.test_id += "_advC"
        if args.use_imcLoss: 
            args.test_id += "_imcL"
        if args.use_sharpFeat: 
            args.test_id += "_s&fL"
        if args.use_clsBal:
            args.test_id += "_clsB"

        args.test_id += "_rs{}".format(args.real_size)

        
        print(">>>>> TEST ID: ", args.test_id)

        if not os.path.exists(os.path.join(args.prefix, args.save_path, args.test_id, args.model_step,)):
            os.makedirs(os.path.join(args.prefix, args.save_path, args.test_id, args.model_step,))

        if not os.path.exists(os.path.join(args.prefix, args.save_path, args.test_id, args.model_step, "images")):
            os.makedirs(os.path.join(args.prefix, args.save_path, args.test_id, args.model_step, "images"))

        if not os.path.exists(os.path.join(args.prefix, args.save_path, args.test_id, args.model_step, "labels")):
            os.makedirs(os.path.join(args.prefix, args.save_path, args.test_id, args.model_step, "labels"))

        return args


class AugGenerator(object):

    def __init__(self, args):
        """
        ### Initialize
        """
        self.args = args


        # $\log_2$ of image resolution
        log_resolution = int(math.log2(self.args.image_size))

        # Create discriminator and generator
        #activation = logTanh()
        #activation = nn.Tanh()

        if self.args.use_genXL: 
            self.generator = MaskGeneratorXL(log_resolution, self.args.d_latent, self.args.num_classes).to(self.args.device)
        else:
            self.generator = MaskGenerator(log_resolution, self.args.d_latent, self.args.num_classes).to(self.args.device)

        #self.generator = MaskGeneratorXL(log_resolution, self.args.d_latent, self.args.num_classes).to(self.args.device)
        self.args.pretrained_G = self.generator.__class__.__name__ + "_ckpt.pt"
        
        
        # Get number of generator blocks for creating style and noise inputs
        self.n_gen_blocks = self.generator.n_blocks
        # Create mapping network
        self.mapping_network = MappingNetwork(self.args.d_latent, self.args.mapping_network_layers).to(self.args.device)
        
        if self.args.useCond:
            self.adapter = Adapter(self.args.d_latent, self.args.d_img_space).to(self.args.device)
            self.feature_extractor = GetFeatureExtractor(model=mobilenet_v3_small()).to(self.args.device)
        
        # Create path length penalty loss
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.args.device)

        #GRADIENT PENALITY
        self.gradient_penalty = GradientPenalty().to(self.args.device)

        dataset = PatchDataset(os.path.join(self.args.prefix,self.args.patch_path), image_size=self.args.image_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.workers,
                                                 shuffle=True, drop_last=True, pin_memory=True)

        self.loader = cycle_dataloader(dataloader)

        self.legenda = {"black":"background", "red":"rigid_plastic", "yellow":"cardboard", "blue":"metal", "purple":"soft_plastic"}

        # CHECKPOINT RESUME

        #generator
        ckpt_file = os.path.join(self.args.prefix, self.args.root, self.args.test_id, self.args.model_step, self.args.pretrained_G)
        checkpoint = torch.load(ckpt_file)
        self.generator.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded GENERATOR from {ckpt_file}")

        #mapping network
        ckpt_file = os.path.join(self.args.prefix, self.args.root, self.args.test_id, self.args.model_step, self.args.pretrained_W)
        checkpoint = torch.load(ckpt_file)

        #print(self.mapping_network.state_dict().keys())

        self.mapping_network.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded MAPPING NETWORK from {ckpt_file}")

        #adapter network
        if self.args.useCond:
            ckpt_file = os.path.join(self.args.prefix, self.args.root, self.args.test_id, self.args.model_step, self.args.pretrained_A)
            checkpoint = torch.load(ckpt_file)
            self.adapter.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded MAPPING NETWORK from {ckpt_file}")
        
    def get_w(self, batch_size: int, cond_img: torch.Tensor = None):
        """
        This samples $z$ randomly and get $w$ from the mapping network.

        We also apply style mixing sometimes where we generate two latent variables
        $z_1$ and $z_2$ and get corresponding $w_1$ and $w_2$.
        Then we randomly sample a cross-over point and apply $w_1$ to
        the generator blocks before the cross-over point and
        $w_2$ to the blocks after.
        """

        # Mix styles
        if torch.rand(()).item() < self.args.style_mixing_prob:
            # Random cross-over point
            cross_over_point = int(torch.rand(()).item() * self.n_gen_blocks)
            # Sample $z_1$ and $z_2$
            z2 = torch.randn(batch_size, self.args.d_latent).to(self.args.device)
            z1 = torch.randn(batch_size, self.args.d_latent).to(self.args.device)
            # Get $w_1$ and $w_2$
            #print("Z1: ", z1.shape)
            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)

            #Add the conditioning informations (maybe we can condition a different level by exctraing differen features)
            if self.args.useCond:
                c = self.feature_extractor(cond_img)
                #print("c: ", c['linear'].shape)
                #print("w: ", w1.shape)
                w1 = self.adapter(w1,c['linear'].squeeze())
                w2 = self.adapter(w2,c['linear'].squeeze())

            # Expand $w_1$ and $w_2$ for the generator blocks and concatenate
            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.n_gen_blocks - cross_over_point, -1, -1)
            w = torch.cat((w1, w2), dim=0)
            return w
            
        # Without mixing
        else:
            # Sample $z$ and $z$
            z = torch.randn(batch_size, self.args.d_latent).to(self.args.device)
            # Get $w$ and $w$
            w = self.mapping_network(z)

            #Add the conditioning informations (maybe we can condition a different level by exctraing differen features)
            if self.args.useCond:
                c = self.feature_extractor(cond_img)
                #print("c: ", c['linear'].shape)
                #print("w: ", w.shape)
                w = self.adapter(w,c['linear'].squeeze())

            # Expand $w$ for the generator blocks
            return w[None, :, :].expand(self.n_gen_blocks, -1, -1) #replicate w for each layer of the network

    def get_noise(self, batch_size: int):
        """
        This generates noise for each [generator block](index.html#generator_block)
        """
        # List to store noise
        noise = []
        # Noise resolution starts from $4$
        resolution = 4

        # Generate noise for each generator block
        for i in range(self.n_gen_blocks):
            # The first block has only one $3 \times 3$ convolution
            if i == 0:
                n1 = None
            # Generate noise to add after the first convolution layer
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=self.args.device)
            # Generate noise to add after the second convolution layer
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=self.args.device)
            n3 = torch.randn(batch_size, 1, resolution, resolution, device=self.args.device) #MOD 

            # Add noise tensors to the list
            noise.append((n1, n2, n3))

            # Next block has $2 \times$ resolution
            resolution *= 2

        # Return noise tensors
        # print("NOISE: ")
        # for i,n in enumerate(noise):
        #     print(f"{i} : {n[1].shape}")
        return noise

    def generate_images(self, batch_size: int, cond_img: torch.Tensor = None):
        """
        This generate images using the generator
        """

        # Get $w$
        w = self.get_w(batch_size, cond_img=cond_img)
        self.std_w = batchSTD(w)
        # Get noise
        noise = self.get_noise(batch_size)

        # Generate images
        images, seg, _ = self.generator(w, noise)

        #seg = torch.argmax(seg, axis=1).unsqueeze(1) / self.args.num_classes
        #print("seg value: ", torch.unique(seg))
        #print("W: ", w.shape)
        #print("noise: ", noise.shape)

        # Return images and $w$
        return images, seg, w

    def saveBatchedIamges(self, gen_imgs, gen_masks, index):

        gen_imgs = normalizeRGB(gen_imgs, use_int8=True)
        gen_imgs = gen_imgs.detach().permute(0,2,3,1).cpu().numpy()
        gen_masks = torch.argmax(gen_masks, dim=1)
        gen_masks = gen_masks.detach().cpu().numpy()

        root_path = os.path.join(self.args.prefix, self.args.save_path, self.args.test_id, self.args.model_step)

        for i in range(gen_imgs.shape[0]):

            img = gen_imgs[i,:,:,:]
        
            #plt.subplot(2,1,1)
            #plt.imshow(img)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(root_path, "images", "augImg_{:06d}.png".format(index)), img)

            mask = gen_masks[i,:,:]

            #plt.subplot(2,1,2)
            #plt.imshow(mask)

            #plt.show()


            cv2.imwrite(os.path.join(self.args.prefix, root_path, "labels", "augImg_{:06d}.png".format(index)), mask)

    def generate(self, args):

        print("STARTING GENERATION IN {}".format(os.path.join(self.args.prefix, self.args.save_path, self.args.test_id, self.args.model_step)))
        for i in range(self.args.dataset_size):

            if i%(self.args.dataset_size//10)==0:
                print(f">>>> Generated {i+1} images")

            patched_img = next(self.loader)
            patched_img = patched_img.to(self.args.device) 
            generated_images, generated_masks, _ = self.generate_images(self.args.batch_size, patched_img)

            self.saveBatchedIamges(generated_images, generated_masks, i)



def main():
    """
    ### Train StyleGAN2
    """
    #CONFIGURATION
    args = sys.argv
    prefix = args[1] if len(args)>1 else "."

    config_file = os.path.join(prefix,"config/generation.yaml")
    print("Loading configuration from ", config_file)
    cfg = Configurator(config_file, prefix)
    args = cfg.getArgs()

    args.categories = ["bg+paper","rigid_plastic", "cardboard", "metal", "soft_plastic"]

    #CREATE TRAINER OBJECT
    gen = AugGenerator(args)

    gen.generate(args) #start training loop


if __name__ == '__main__':
    main()

