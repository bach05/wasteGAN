"""
We trained this on [CelebA-HQ dataset](https://github.com/tkarras/progressive_growing_of_gans).
You can find the download instruction in this
[discussion on fast.ai](https://forums.fast.ai/t/download-celeba-hq-dataset/45873/3).
Save the images inside [`data/stylegan` folder](#dataset_path).
"""
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
from util.losses import *
#from swd_pytorch.swd import swd
#from geomloss import SamplesLoss
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

#from labml import tracker, lab, monit, experiment
#from labml.configs import BaseConfigs
#from labml_helpers.device import DeviceConfigs
#from labml_helpers.train_valid import ModeState, hook_model_outputs
from models.styleGAN2ada import MaskDiscriminator, MaskGenerator, MappingNetwork, GradientPenalty, PathLengthPenalty, BranchedMaskDiscriminator, MaskGeneratorXL, FeatMaskGenerator, PatchedMaskDiscriminator
from models.styleGAN2ada import logTanh, Sloss, Adapter, GetFeatureExtractor
from util.GANlosses import DiscriminatorLoss, GeneratorLoss, GeneratorLoss2, DiscriminatorLoss2, GeneratorLoss3
from util.styleGAN import *
from munch import DefaultMunch
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from util.diffusion_utils import *
from util.customTransforms import *
from util.condDataset import MaskGANDataset
from tqdm import tqdm
from util.augment import AugmentPipe
import matplotlib.pyplot as plt
import time
import logging 
from util.condDataset import condDataset, customCondDataset, MaskGANDatasetWithPatch
import copy
from util.externalTools import computeAvgGrad, batchSTD, prep_dseg_output
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from utils.util import generate_distinguishable_colors


augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, gamma=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, gamma=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, gamma=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=0, rotate90=0, xint=0, scale=0, rotate=0, aniso=0, xfrac=0, brightness=1, contrast=1, gamma=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'pint':  dict(brightness=1, contrast=1, gamma=1, hue=1, saturation=1, sharpness=1, noise=1),
        'bgcfnc': dict(xflip=0, rotate90=0, xint=0, scale=0, rotate=0, aniso=0, xfrac=0, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

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
            args.train_id += "_genXL"
        if args.use_genFeat: 
            args.train_id += "_genFeat"
        if args.use_symLogAct: 
            args.train_id += "_symLogA{}".format(args.symLogAct_a)        
        if args.use_adaCustom: 
            args.train_id += "_adaC"
        if args.use_advCustom: 
            args.train_id += "_advC"
        if args.use_imcLoss: 
            args.train_id += "_imcL"
        if args.use_sharpFeat: 
            args.train_id += "_s&fL"
        if args.use_clsBal:
            args.train_id += "_clsB"
        if args.use_PatchDisc:
            args.train_id += "_patch"

        print(">>>>> TRAIN ID: ", args.train_id)

        args.train_id += "_rs{}".format(args.real_size)

        if not os.path.exists(os.path.join(args.prefix, args.model_path,args.train_id)):
            os.makedirs(os.path.join(args.prefix, args.model_path,args.train_id))
            print(f"Created folder: {os.path.join(args.prefix, args.model_path,args.train_id)}")

        #LOGGING
        logging.basicConfig(filename=os.path.join(args.prefix, args.model_path,args.train_id,f"log.std"), 
					format='%(asctime)s %(message)s', 
					filemode='w') 

        #Let us Create an object 
        args.file_log=logging.getLogger() 

        #Now we are going to Set the threshold of logger to DEBUG 
        args.file_log.setLevel(logging.DEBUG)

        return args
    

class Trainer(object):

    def __init__(self, args):
        """
        ### Initialize
        """
        self.args = args

        self.logger = SummaryWriter(os.path.join(self.args.prefix, self.args.tensorboard_path,self.args.train_id))
        self.avg_log = {"avg_loss_D":0, "avg_loss_D_rgb":0, "avg_loss_D_seg":0, "avg_loss_G":0, "avg_out_D_fake":0, "avg_out_D_fake_rgb":0, "avg_out_D_fake_seg":0, 
                        "avg_out_D_real":0, "avg_out_D_real_rgb":0, "avg_out_D_real_seg":0, "avg_adv_G_loss":0, "avg_feat_G_loss":0, "avg_recon_G_loss":0, 
                        "avg_sharp_G_loss":0, "avg_clsB_G_loss":0}

        ### VISUALIZATION HELPERS
        self.permute_list = []
        for i in range(self.args.batch_size):
                self.permute_list.append(i)
                self.permute_list.append(self.args.batch_size+i)
        self.permute_list = torch.Tensor(self.permute_list).to(torch.long)
        #print("PERMUTATION LISTFOR VISUALIZATION: ", self.permute_list)

        print("***** TRAIN ID: ", args.train_id)

        # Create dataset
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
        train_transform = MultiCompose([MultiResize(self.args.image_size*2), MultiRandomCrop(self.args.image_size), MultiHorizontalFlip(), MultiToTensor(), MultiNormalize(mean, std, mask=1)])
        dataset = MaskGANDatasetWithPatch(os.path.join(self.args.prefix,self.args.image_path), os.path.join(self.args.prefix,self.args.mask_path), os.path.join(self.args.prefix,self.args.patch_path), image_size=self.args.image_size, n_classes=self.args.num_classes, transforms=train_transform, size=args.real_size)
        #dataset = MaskGANDataset(self.args.image_path,self.args.mask_path, image_size=self.args.image_size, n_classes=self.args.num_classes, transforms=train_transform)
        
        val_transform = MultiCompose([MultiResize(self.args.image_size*2), MultiRandomCrop(self.args.image_size, mode='center'), MultiToTensor(), MultiNormalize(mean, std, mask=1)])
        dataset_val = MaskGANDatasetWithPatch(os.path.join(self.args.prefix,self.args.image_path_val), os.path.join(self.args.prefix,self.args.mask_path_val), os.path.join(self.args.prefix,self.args.patch_path), image_size=self.args.image_size, n_classes=self.args.num_classes, transforms=val_transform, size=240)
        #dataset_val = MaskGANDataset(self.args.image_path_val, self.args.mask_path_val, image_size=self.args.image_size, n_classes=self.args.num_classes, transforms=val_transform, size=240)

        # Create data loader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.workers,
                                                 shuffle=True, drop_last=True, pin_memory=True)
        # Continuous [cyclic loader](../../utils.html#cycle_dataloader)
        self.pretrain_loader = copy.deepcopy(dataloader)
        self.loader = cycle_dataloader(dataloader)

        self.dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=self.args.batch_size,
                                         shuffle=False, num_workers=2)

        # $\log_2$ of image resolution
        log_resolution = int(math.log2(self.args.image_size))

        # Create discriminator and generator
        if self.args.use_symLogAct:
            activation = logTanh(a=self.args.use_symLogAct)
        else:
            activation = None

        if args.use_PatchDisc:
            self.discriminator_rgb = PatchedMaskDiscriminator(log_resolution, 3, n_features=64, max_features=512,
                                                       useActivation=activation).to(self.args.device)
            self.discriminator_seg = PatchedMaskDiscriminator(log_resolution, args.num_classes + 3, n_features=8,
                                                       max_features=32, useActivation=activation).to(self.args.device)
        else:
            self.discriminator_rgb = MaskDiscriminator(log_resolution, 3, n_features=64, max_features=512, useActivation=activation).to(self.args.device)
            self.discriminator_seg = MaskDiscriminator(log_resolution, args.num_classes+3, n_features=8, max_features=32, useActivation=activation).to(self.args.device)
        if self.args.use_genXL: 
            self.generator = MaskGeneratorXL(log_resolution, self.args.d_latent, self.args.num_classes).to(self.args.device)
        elif self.args.use_genFeat:
            self.generator = FeatMaskGenerator([2048,1024,512,256,64,3], self.args.d_latent, self.args.num_classes).to(self.args.device)
        else:
            self.generator = MaskGenerator(log_resolution, self.args.d_latent, self.args.num_classes).to(self.args.device)
        
        # Get number of generator blocks for creating style and noise inputs
        self.n_gen_blocks = self.generator.n_blocks
        # Create mapping network
        self.mapping_network = MappingNetwork(self.args.d_latent, self.args.mapping_network_layers).to(self.args.device)
        
        if self.args.useCond:
            self.adapter = Adapter(self.args.d_latent, self.args.d_img_space).to(self.args.device)
            self.feature_extractor = GetFeatureExtractor(model=mobilenet_v3_small()).to(self.args.device)

        # Create path length penalty loss
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.args.device)

        # input = torch.rand((12, 3, 256, 256)).to(self.args.device)
        # self.logger.add_graph(self.discriminator_rgb, input_to_model=input, verbose=False)
        # input = torch.rand((12, 1, 256, 256)).to(self.args.device)
        # self.logger.add_graph(self.discriminator_seg, input_to_model=input, verbose=False)

        # input = torch.rand((7, 12, 256)).to(self.args.device)
        # noise = self.get_noise(12)
        # self.logger.add_graph(self.generator, input_to_model=None, verbose=False)

        # input = torch.rand((12, 256)).to(self.args.device)
        # self.logger.add_graph(self.mapping_network, input_to_model=input, verbose=False)

        
        #print(summary(self.discriminator, (12, 4, 256, 256)))
        #print(summary(self.generator, (7, 12, 256)))
        #print(summary(self.mapping_network, (12, 256)))
        
        # Add model hooks to monitor layer outputs
        # if self.log_layer_outputs:
        #     hook_model_outputs(self.mode, self.discriminator, 'discriminator')
        #     hook_model_outputs(self.mode, self.generator, 'generator')
        #     hook_model_outputs(self.mode, self.mapping_network, 'mapping_network')

        #GRADIENT PENALiTY
        self.gradient_penalty = GradientPenalty().to(self.args.device)

        # Discriminator and generator losses
        if self.args.use_advCustom:
            self.discriminator_loss = DiscriminatorLoss2().to(self.args.device)
            self.generator_loss =  GeneratorLoss3().to(self.args.device)
        else:
            self.discriminator_loss = DiscriminatorLoss().to(self.args.device)
            self.generator_loss = GeneratorLoss().to(self.args.device)
        
        #self.discriminator_loss = DiscriminatorLoss().to(self.args.device)
        #self.discriminator_loss2 = DiscriminatorLoss2().to(self.args.device)
        #self.generator_loss = GeneratorLoss().to(self.args.device)
        #self.generator_loss3 = GeneratorLoss3().to(self.args.device)

        #self.loss_fn_vgg = lpips.LPIPS(net='alex').to(self.args.device) # perceptual loss
        #self.loss_fn_vgg = resnetPercLoss("./output_models/semSeg_ep199_.pth", self.args.device)
        hub_dir = os.path.join(self.args.prefix, self.args.hub_dir)
        torch.hub.set_dir(hub_dir)
        print("HUB DIR: ", torch.hub.get_dir())
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(self.args.device)
        self.mse = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        #self.smooth_l1loss = nn.SmoothL1Loss()
       
        #self.mse = nn.MSELoss()
        #self.cel = nn.CrossEntropyLoss()
        input_shape = [self.args.batch_size, self.args.image_size*self.args.image_size, 3]
        #self.mi = MutualInformation(in_shape=input_shape, sigma=0.1, normalize=True, device=self.args.device).to(self.args.device)
        self.used_distribution = self.args.distribution
        self.mi2_f = MutualInformation2(self.args.num_classes, device=self.args.device, fig_name=self.used_distribution, cat=self.args.categories, use_ema=self.args.useEMA, ema_w=self.args.alpha).to(self.args.device)
        self.mi2_r = MutualInformation2(self.args.num_classes, device=self.args.device, fig_name=self.used_distribution, cat=self.args.categories, use_ema=self.args.useEMA, ema_w=self.args.alpha).to(self.args.device)
        #self.WSloss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
        #self.entropy = Entropy(device = self.args.device)
        self.bhattacharyya = Bhattacharyya()
        self.bhattacharyya2 = BhattacharyyaMgI()
        self.bhattacharyya3 = BhattacharyyaIgM()
        self.sharpLoss = SharpLoss(device=self.args.device)
        self.classBalance = ClassBalance(self.args.num_classes, device=self.args.device)

        self.FID = FIDscore(device=self.args.device)

        self.legenda = {"black":"background","red":"rigid_plastic", "yellow":"cardboard", "blue":"metal", "purple":"soft_plastic"}
        self.logger.add_text("[LEGENDA]", str(self.legenda), global_step=0)

        # Create optimizers
        self.discriminator_optimizer_rgb = torch.optim.Adam(
            self.discriminator_rgb.parameters(),
            lr=self.args.dis_learning_rate_rgb, betas=self.args.adam_betas
        )
        self.discriminator_optimizer_seg = torch.optim.Adam(
            self.discriminator_seg.parameters(),
            lr=self.args.dis_learning_rate_seg, betas=self.args.adam_betas
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.args.gen_learning_rate, betas=self.args.adam_betas
        )
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=self.args.mapping_network_learning_rate, betas=self.args.adam_betas
        )
        if self.args.useCond:
            self.adapter_optimizer = torch.optim.Adam(
                self.adapter.parameters(),
                lr=self.args.mapping_network_learning_rate, betas=self.args.adam_betas
            )

        # PRETRAINED MODEL LOADING
        if self.args.pretrained_init: 

            #discriminator
            ckpt_file = os.path.join(self.args.prefix, self.args.root, self.args.pretrained_D)
            checkpoint = torch.load(ckpt_file)
            self.discriminator_rgb.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded DISCRIMINATOR from {ckpt_file}")

            #generator
            ckpt_file = os.path.join(self.args.prefix, self.args.root, self.args.pretrained_G)
            checkpoint = torch.load(ckpt_file)
            self.generator.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded GENERATOR from {ckpt_file}")

            #mapping network
            ckpt_file = os.path.join(self.args.prefix, self.args.root, self.args.pretrained_W)
            checkpoint = torch.load(ckpt_file)
            self.mapping_network.load_state_dict(checkpoint['model_state_dict'])
        # CHECKPOINT RESUME
        self.ckpt_iter = 0
        init_p = self.args.init_p
        if self.args.load_checkpoint: 
            ckpt_folder = os.path.join(self.args.prefix, self.args.model_path, self.args.train_id, f"step_{self.args.check_point_step}")
            ckpt_list = os.listdir(ckpt_folder)
            
            for ckpt in ckpt_list:
                checkpoint = torch.load(os.path.join(self.args.prefix, ckpt_folder, ckpt))
                self.ckpt_iter = checkpoint['epoch']
                init_p = checkpoint['p']

                if ckpt[:-8] == self.discriminator.__class__.__name__:
                    self.discriminator.load_state_dict(checkpoint['model_state_dict'])
                    self.discriminator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    #loss = checkpoint['loss']
                    print(f"Loaded DISCRIMINATOR from {ckpt}")

                if ckpt[:-8] == self.generator.__class__.__name__:
                    self.generator.load_state_dict(checkpoint['model_state_dict'])
                    self.generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.generator.train()
                    print(f"Loaded GENERATOR from {ckpt}")

                if ckpt[:-8] == self.mapping_network.__class__.__name__:
                    self.mapping_network.load_state_dict(checkpoint['model_state_dict'])
                    self.mapping_network_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.mapping_network.train()
                    print(f"Loaded MAPPING NET from {ckpt}")


        print(f"Init ADA with p={init_p}")
        if self.args.use_adaCustom:
            self.p_sched = pSchedulerCustom(target_acc=self.args.target_acc, init_p=init_p, integration_steps=self.args.integration_steps)
        else: 
            self.p_sched = pScheduler(target_acc=self.args.target_acc, init_p=init_p, integration_steps=self.args.integration_steps)

        
        self.p, self.avg_acc, self.avg_diff = self.p_sched.get_p()
        self.augmenter = AugmentPipe(init_p=self.p, **augpipe_specs['bgcfnc']).requires_grad_(False).to(self.args.device)


        # Set tracker configurations
        #tracker.set_image("generated", True)
        
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
        images, seg, internal_state = self.generator(w, noise)

        # for i, int_st in enumerate(internal_state):
        #     print("Output Feat at level {}: {}".format(i , int_st[2].shape) )

        #seg = torch.argmax(seg, axis=1).unsqueeze(1) / self.args.num_classes
        #print("seg value: ", torch.unique(seg))
        #print("W: ", w.shape)
        #print("noise: ", noise.shape)

        # Return images and $w$
        return images, seg, w, internal_state

    def getPatchedImage(self, image, mask):

        mask_flat = torch.sum(mask, dim=1).unsqueeze(1).repeat(1,3,1,1)
        #print(mask_flat.shape)

        patched_image = torch.zeros_like(image)
        patched_image[mask_flat != 0] = image[mask_flat !=0]

        return patched_image     


    def step(self, idx: int):
        """
        ### Training Step
        """

        self.args.idx = idx

        ###################################à
        # Train the DISCRIMINATOR
        #################################

        # Reset gradients
        self.discriminator_optimizer_rgb.zero_grad()
        self.discriminator_optimizer_seg.zero_grad()

        # Accumulate gradients for `gradient_accumulate_steps`
        for i in range(self.args.gradient_accumulate_steps):
            
            if self.args.use_ada and idx % self.args.p_update_interval == 0:
                self.p, self.avg_acc, self.avg_diff = self.p_sched.get_p() 
                self.augmenter.update_p(self.p)
                self.logger.add_scalar("[TRAIN] ADA PROB (p)", self.p, global_step=idx)
                self.logger.add_scalars("[TRAIN] PSCHED LOG",{f"AVG ACC (target {self.args.target_acc})": self.avg_acc, f"AVG ERR": self.avg_acc - self.args.target_acc, "AVG DIFF": self.avg_diff}, global_step=idx)

            # Get real images from the data loader
            real_images, real_mask, patch_image = next(self.loader)
            #print(f"REAL IMG Min {np.min(real_images[0,0].detach().cpu().numpy())} - Max {np.max(real_images[0,0].detach().cpu().numpy())}")
            real_images = real_images.to(self.args.device)
            real_mask = real_mask.to(self.args.device) 
            patch_image = patch_image.to(self.args.device) 

            #APPLY ADA
            #plt.subplot(2,1,1)
            #plot_grid_images(real_images, nrow=8, normalize=True)
            real_images_original = real_images.clone()
            if self.args.use_ada:
                self.args.debug_str = "DIC real images"
                real_images, _ = self.augmenter(real_images, args=self.args, debug=False)
                self.nan_flag = debug_NaN("AUGMENTED Real", real_images, self.logger, idx,  self.args.file_log)
                #real_images.require_grad = True
            #plt.subplot(2,1,2)
            #plot_grid_images(real_images, nrow=8, normalize=True)
            #plt.show()

            # Sample images from generator
            generated_images, generated_masks, _, int_state = self.generate_images(self.args.batch_size, patch_image)
            #print(f"GEN IMG Min {np.min(generated_images[0,0].detach().cpu().numpy())} - Max {np.max(generated_images[0,0].detach().cpu().numpy())}")

            self.nan_flag = debug_NaN("Generated (DIS)", generated_images, self.logger, idx,  self.args.file_log)

            #Apply ADA
            generated_images_original = generated_images.clone()
            if self.args.use_ada:
                self.args.debug_str = "DIC gen images"
                generated_images, _ = self.augmenter(generated_images, args=self.args, debug=False)
                self.nan_flag = debug_NaN("AUGMENTED Generated (DIS)", generated_images, self.logger, idx,  self.args.file_log)

                    #print("grad of gen images: ", generated_images.requires_grad)
                #generated_images.requires_grad = True
            # Discriminator classification for generated images
            #print(f"[GEN] Image {generated_images.shape} , Mask {generated_masks.shape} ")
            #generated_data = torch.cat((generated_images, generated_masks), dim=1)
            #print("generated data: ", generated_data.shape)

            fake_output_rgb = self.discriminator_rgb(generated_images.detach()) #why detach here??? avoid backwards problems
            input_dis_seg = torch.cat([generated_masks, generated_images], dim=1)
            fake_output_seg = self.discriminator_seg(input_dis_seg.detach()) #why detach here??? avoid backwards problems
            self.avg_log["avg_out_D_fake_rgb"] += torch.mean(fake_output_rgb).item()
            self.avg_log["avg_out_D_fake_seg"] += torch.mean(fake_output_seg).item()

            self.nan_flag = debug_NaN("Real", real_images, self.logger, idx,  self.args.file_log)

            # We need to calculate gradients w.r.t. real images for gradient penalty
            if self.args.use_gp and ((idx + 1) % self.args.lazy_gradient_penalty_interval == 0):
                real_images.requires_grad_()
                real_mask.requires_grad_()
            # Discriminator classification for real images
            #real_data = torch.cat((real_images,real_mask), dim=1)
            real_output_rgb = self.discriminator_rgb(real_images)
            input_dis_seg = torch.cat([real_mask, real_images], dim=1)
            real_output_seg = self.discriminator_seg(input_dis_seg)
            self.avg_log["avg_out_D_real_rgb"] += torch.mean(real_output_rgb).item()
            self.avg_log["avg_out_D_real_seg"] += torch.mean(real_output_seg).item()

            if self.args.use_ada:
                self.p_sched.step((real_output_rgb), (fake_output_rgb)) #update the p scheduler

            # Get discriminator loss

            #real_loss_rgb, fake_loss_rgb = self.discriminator_loss2(real_output_rgb, fake_output_rgb)
            #real_loss_seg, fake_loss_seg = self.discriminator_loss2(real_output_seg, fake_output_seg)

            real_loss_rgb, fake_loss_rgb = self.discriminator_loss(real_output_rgb, fake_output_rgb)
            real_loss_seg, fake_loss_seg = self.discriminator_loss(real_output_seg, fake_output_seg)
            
            #feat_loss = self.lpips(normalizeRGB(generated_images_original), real_images_original)
            
            disc_loss_rgb = (real_loss_rgb + fake_loss_rgb) #+ feat_loss
            disc_loss_seg = (real_loss_seg + fake_loss_seg)

            # Add gradient penalty
            if self.args.use_gp and ( (idx + 1) % self.args.lazy_gradient_penalty_interval == 0):
                # Calculate and log gradient penalty
                gp_rgb = self.gradient_penalty(real_images, real_output_rgb)
                gp_seg = self.gradient_penalty(real_mask, real_output_seg)
                # Multiply by coefficient and add gradient penalty
                disc_loss_rgb = disc_loss_rgb + 0.5 * self.args.gradient_penalty_coefficient * gp_rgb * self.args.lazy_gradient_penalty_interval
                disc_loss_seg = disc_loss_seg + 0.5 * self.args.gradient_penalty_coefficient * gp_seg * self.args.lazy_gradient_penalty_interval


            # Log discriminator loss
            self.logger.add_scalars("[TRAIN][DISC] LOSS", {"RGB": disc_loss_rgb.item(), "SEG": disc_loss_seg.item()}, global_step=idx)
            self.logger.add_scalars("[TRAIN][DISC] OUT", {"RGB REAL": torch.mean(real_output_rgb), "RGB FAKE": torch.mean(fake_output_rgb), "SEG REAL": torch.mean(real_output_seg), "SEG FAKE": torch.mean(fake_output_seg)}, global_step=idx)
            self.logger.add_scalars("[TRAIN][DISC] RGB&SEG LOSS", {"REAL RGB": real_loss_rgb, "REAL SEG": real_loss_seg, "FAKE RGB": fake_loss_rgb, "FAKE SEG": fake_loss_seg}, global_step=idx)
            
            
            self.avg_log["avg_loss_D_rgb"] += disc_loss_rgb.item()
            self.avg_log["avg_loss_D_seg"] += disc_loss_seg.item()

            #self.logger.add_scalars("[TRAIN][DISC] OUTPUT", {'REAL': torch.mean(real_output).item(), 'FAKE': torch.mean(fake_output).item()}, global_step=idx)

            if (idx + 1) % self.args.log_generated_interval == 0:
                # Log discriminator model parameters occasionally
                self.logger.add_scalars("[TRAIN][DISC] LOSS AVG", {"RGB": self.avg_log["avg_loss_D_rgb"] / self.args.log_generated_interval, "SEG": self.avg_log["avg_loss_D_seg"] / self.args.log_generated_interval}, global_step=idx)
                self.avg_log["avg_loss_D_rgb"] = 0
                self.avg_log["avg_loss_D_seg"] = 0

                ######### VALIDATION #######
                ############################
                self.discriminator_rgb.eval()
                self.discriminator_seg.eval()
                self.avg_log["avg_out_D_val_rgb"] = 0
                self.avg_log["avg_out_D_val_seg"] = 0

                for j, data in enumerate(self.dataloader_val):
                    real_images_val, mask_val, _ = data
                    real_images_val = real_images_val.to(self.args.device)
                    mask_val = mask_val.to(self.args.device)
                    #print("val img: ", real_images_val.shape)
                    #val_real_data = torch.cat((real_images_val, mask_val), dim=1)
                    val_output_rgb = self.discriminator_rgb(real_images_val)
                    input_dis_seg = torch.cat([mask_val, real_images_val], dim=1)
                    val_output_seg = self.discriminator_seg(input_dis_seg)
                    self.avg_log["avg_out_D_val_rgb"] += torch.mean(val_output_rgb).item()
                    self.avg_log["avg_out_D_val_seg"] += torch.mean(val_output_seg).item()

                self.discriminator_rgb.train()
                self.discriminator_seg.train()

                self.logger.add_scalars("[VAL] Disc Out Avg", {'VAL RGB': self.avg_log["avg_out_D_val_rgb"]/self.args.log_generated_interval, "VAL SEG":self.avg_log["avg_out_D_val_seg"]/self.args.log_generated_interval, 'REAL RGB': self.avg_log["avg_out_D_real_rgb"]/self.args.log_generated_interval,  'FAKE RGB': self.avg_log["avg_out_D_fake_rgb"]/self.args.log_generated_interval, 'REAL SEG': self.avg_log["avg_out_D_real_seg"]/self.args.log_generated_interval,  'FAKE SEG': self.avg_log["avg_out_D_fake_seg"]/self.args.log_generated_interval}, global_step=idx)
                self.avg_log["avg_out_D_val"] = 0
                self.avg_log["avg_out_D_val_rgb"] = 0
                self.avg_log["avg_out_D_val_seg"] = 0
                self.avg_log["avg_out_D_fake"] = 0
                self.avg_log["avg_out_D_real"] = 0
                self.avg_log["avg_out_D_real_rgb"] = 0
                self.avg_log["avg_out_D_fake_rgb"] = 0
                self.avg_log["avg_out_D_real_seg"] = 0
                self.avg_log["avg_out_D_fake_seg"] = 0
                


            # Compute gradients
            disc_loss_rgb.backward()
            disc_loss_seg.backward()

            #check grad
            grads_avg_D_rgb = computeAvgGrad(self.discriminator_rgb)

            # Clip gradients for stabilization
            torch.nn.utils.clip_grad_norm_(self.discriminator_rgb.parameters(), max_norm=1.0)
            
            #print("init")
            #start = time.time()
            
            # Take optimizer step
            self.discriminator_optimizer_rgb.step()
            

            #check grad
            grads_avg_D_seg = computeAvgGrad(self.discriminator_seg)

            # Clip gradients for stabilization
            torch.nn.utils.clip_grad_norm_(self.discriminator_seg.parameters(), max_norm=1.0)
            
            #print("init")
            #start = time.time()
            
            # Take optimizer step
            self.discriminator_optimizer_seg.step()

            #end = time.time()
            #print("end: ", end-start)

        #########################################
        # Train the GENERATOR
        #########################################

        # Reset gradients
        self.generator_optimizer.zero_grad()
        self.mapping_network_optimizer.zero_grad()

        # Accumulate gradients for `gradient_accumulate_steps`
        for i in range(self.args.gradient_accumulate_steps):
            # Sample images from generator
            generated_images, generated_masks, w, int_state = self.generate_images(self.args.batch_size,  patch_image)
            generated_images_original = generated_images.clone()
            self.nan_flag = debug_NaN("Generated (GEN)", generated_images, self.logger, idx,  self.args.file_log)
            self.nan_flag = debug_NaN("W", w, self.logger, idx,  self.args.file_log)
            #APPLY ADA
            if self.args.use_ada:
                self.args.debug_str = "GEN gen images"
                generated_images, _ = self.augmenter(generated_images, args=self.args, debug=False) #--> lead generator to produce augmented images

                self.nan_flag = debug_NaN("Augmented Generated (GEN)", generated_images, self.logger, idx,  self.args.file_log)
                #generated_images.requires_grad = True

            # Discriminator classification for generated images
            #generated_data = torch.cat((generated_images, generated_masks), dim=1)
            fake_output_rgb = self.discriminator_rgb(generated_images)
            input_dis_seg = torch.cat([generated_masks, real_images], dim=1)
            fake_output_seg = self.discriminator_seg(input_dis_seg) #stop grad

            real_output_rgb = self.discriminator_rgb(real_images)
            input_dis_seg = torch.cat([real_mask, real_images], dim=1)
            real_output_seg = self.discriminator_seg(input_dis_seg) #stop gradient

            #fake_output = prep_dseg_output(self.args, fake_output, use_feat=True)
            #real_output = prep_dseg_output(self.args, real_output, use_feat=False)


            # Get generator loss
            #adv_loss_rgb = self.generator_loss3(real_output_rgb, fake_output_rgb)
            #adv_loss_seg = self.generator_loss3(real_output_seg, fake_output_seg)            
            
            if self.args.use_advCustom:
                adv_loss_rgb = self.generator_loss(real_output_rgb, fake_output_rgb)
                adv_loss_seg = self.generator_loss(real_output_seg, fake_output_seg)
            else:
                adv_loss_rgb = self.generator_loss(fake_output_rgb)
                adv_loss_seg = self.generator_loss(fake_output_seg)

            adv_loss = self.args.rgb_w*adv_loss_rgb + self.args.seg_w*adv_loss_seg
            
            #feat_loss_rgb = 0.1* torch.pow(torch.mean(fake_output_rgb-real_output_rgb), 4) #in equilibrium the discriminator is not able to distinguish fake and real sample, so the outup should be similar
            #feat_loss_seg = 0.1* torch.pow(torch.mean(fake_output_seg-real_output_seg), 4)
            #feat_loss = self.args.rgb_w*feat_loss_rgb + self.args.seg_w*feat_loss_seg
            feat_loss = self.lpips(normalizeRGB(generated_images_original), real_images_original)
            
            #recon_loss = self.mse(real_images_original, generated_images)
            # real_mask_rep = real_mask.repeat(1,3,1,1)
            
            # mi1_real = self.mi(real_mask_rep, real_images_original)
            mi2_real = self.mi2_r(real_images_original, real_mask, rgb=True)
            # #print(f"[REAL] M1: {mi1_real}, M2 {mi2_real}")

            # fake_mask_rep = generated_masks.repeat(1,3,1,1)
            # mi1_fake = self.mi(fake_mask_rep, generated_images_original)
            mi2_fake = self.mi2_f(generated_images_original, generated_masks, rgb=True)
            # #print(f"[FAKE] M1: {mi1_fake}, M2 {mi2_fake}")

            # #print(f"M1 dist: {mi1_loss}, M2 dist {mi2_loss}")

            # ws_loss = self.WSloss(mi2_real["pJoint"], mi2_fake["pJoint"])

            # entropy_loss = self.l1loss(self.entropy(mi2_real["pJoint"]),  self.entropy(mi2_fake["pJoint"]))

            if self.used_distribution == "pJoint":
                #bhat_loss = self.bhattacharyya(mi2_real["pJoint"], mi2_fake["pJoint"], rgb=True)
                metric_loss = self.l1loss(mi2_real["pJoint"], mi2_fake["pJoint"])
            if self.used_distribution == "pImg|Mask":
                #bhat_loss = self.bhattacharyya3(mi2_real["pImg|Mask"], mi2_fake["pImg|Mask"], rgb=True)
                metric_loss = self.l1loss(mi2_real["pImg|Mask"], mi2_fake["pImg|Mask"])

            #bhat_loss = self.bhattacharyya(mi2_real["pJoint"], mi2_fake["pJoint"], rgb=True)
            #bhat_loss = self.bhattacharyya2(mi2_real["pMask|Img"], mi2_fake["pMask|Img"], rgb=True)
            #bhat_loss = self.bhattacharyya3(mi2_real["pImg|Mask"], mi2_fake["pImg|Mask"], rgb=True)
            sharp_loss = self.sharpLoss(generated_images_original, real_images_original)

            recon_loss = metric_loss

            cls_balance_loss, class_hist = self.classBalance(generated_masks)

            #gen_loss = self.args.adv_loss_w * adv_loss + self.args.feat_loss_w * feat_loss + recon_loss * self.args.r_loss_w  + sharp_loss * self.args.s_loss_w #full 
            gen_loss = self.args.adv_loss_w * adv_loss #only adv
            #gen_loss = self.args.adv_loss_w * adv_loss + recon_loss * self.args.r_loss_w #no sharp and feat
            #gen_loss = self.args.adv_loss_w * adv_loss + self.args.feat_loss_w * feat_loss + sharp_loss * self.args.s_loss_w #no reconloss
            
            if self.args.use_imcLoss:
                gen_loss += recon_loss * self.args.r_loss_w
            if self.args.use_sharpFeat:
                gen_loss += self.args.feat_loss_w * feat_loss + sharp_loss * self.args.s_loss_w
            if self.args.use_clsBal:
                gen_loss += self.args.clsB_loss_w * cls_balance_loss

            # Add path length penalty
            if idx > self.args.lazy_path_penalty_after and (idx + 1) % self.args.lazy_path_penalty_interval == 0:
                # Calculate path length penalty
                plp = self.path_length_penalty(w, generated_images)
                #self.args.file_log.debug(f"Applied PLP: {plp}")
                # Ignore if `nan`
                if not torch.isnan(plp):
                    #tracker.add('loss.plp', plp)
                    gen_loss = gen_loss + plp
                    self.logger.add_scalar("[TRAIN][GEN] PLP", plp.item(), global_step=idx)


            # Calculate gradients
            gen_loss.backward()

            #check grads
            grads_avg_G = computeAvgGrad(self.generator)
            grads_avg_W = computeAvgGrad(self.mapping_network)
            if self.args.useCond:
                grads_avg_A = computeAvgGrad(self.adapter)

            # Log generator loss
            #tracker.add('loss.generator', gen_loss)
            self.logger.add_scalar("[TRAIN][GEN] LOSS", gen_loss.item(), global_step=idx)
            self.avg_log["avg_loss_G"] += gen_loss.item()

            #RGB AND SEG LOSS LOG
            self.logger.add_scalars("[TRAIN][GEN] RGB&SEG LOSS", {"ADV RGB": adv_loss_rgb, "ADV SEG": adv_loss_seg}, global_step=idx)
            self.logger.add_scalars("[TRAIN][GEN] OUT", {"RGB REAL": torch.mean(real_output_rgb), "RGB FAKE": torch.mean(fake_output_rgb), "SEG REAL": torch.mean(real_output_seg), "SEG FAKE": torch.mean(fake_output_seg)}, global_step=idx)
            
            
            #RECONSTRUCTION LOSS LOG
            self.logger.add_scalars("[TRAIN][GEN] Partial LOSS",
                                    {"ADV LOSS":self.args.adv_loss_w*adv_loss.item(),
                                     "FEAT LOSS":feat_loss.item()*self.args.feat_loss_w,
                                     "REC LOSS": recon_loss.item() * self.args.r_loss_w,
                                     "SHARP LOSS": sharp_loss.item() * self.args.s_loss_w,
                                     "CLS_BAL LOSS": cls_balance_loss.item() * self.args.clsB_loss_w,
                                     }
                                    , global_step=idx)
            self.avg_log["avg_adv_G_loss"] += adv_loss.item()*self.args.adv_loss_w
            self.avg_log["avg_feat_G_loss"] += feat_loss.item()*self.args.feat_loss_w
            self.avg_log["avg_recon_G_loss"] += recon_loss.item()*self.args.r_loss_w
            self.avg_log["avg_sharp_G_loss"] += sharp_loss.item()*self.args.s_loss_w
            self.avg_log["avg_clsB_G_loss"] += cls_balance_loss.item() * self.args.clsB_loss_w
            #self.logger.add_scalars("[TRAIN][GEN] MI LOSS", {"E":entropy_loss,  "W":ws_loss, "B": bhat_loss}, global_step=idx)

          
            #GRADIANT LOG
            if self.args.useCond:
                self.logger.add_scalars("[TRAIN] Average Gradient / Iter", {'G': grads_avg_G,'D RGB': grads_avg_D_rgb, 'D SEG': grads_avg_D_seg, "W": grads_avg_W, "A":grads_avg_A}, idx)
            else:
                self.logger.add_scalars("[TRAIN] Average Gradient / Iter", {'G': grads_avg_G,'D RGB': grads_avg_D_rgb, 'D SEG': grads_avg_D_seg, "W": grads_avg_W}, idx)
            self.logger.add_scalars("[TRAIN] AVG STD / Iter", {'GEN': batchSTD(generated_images_original), "W": self.std_w}, idx)


            if (idx + 1) % self.args.log_generated_interval == 0:
                # Log discriminator model parameters occasionally
                #tracker.add('generator', self.generator)
                #tracker.add('mapping_network', self.mapping_network)
                #step = (idx - self.log_generated_interval) // self.log_generated_interval

                #formatted_strings = [f'{value:.3f}' for value in class_hist]
                #print(f"[LOG] \nClass hist: {formatted_strings}\nclsBal Loss: {cls_balance_loss}")

                self.logger.add_scalar("[TRAIN][GEN] LOSS AVG", self.avg_log["avg_loss_G"] / self.args.log_generated_interval, global_step=idx)
                self.avg_log["avg_loss_G"] = 0

                #RECONSTRUCTION LOSS LOG
                self.logger.add_scalars("[TRAIN][GEN] Partial LOSS AVG",
                                        {"ADV LOSS":self.avg_log["avg_adv_G_loss"] / self.args.log_generated_interval,
                                         "FEAT LOSS": self.avg_log["avg_feat_G_loss"] / self.args.log_generated_interval,
                                         "REC LOSS": self.avg_log["avg_recon_G_loss"] / self.args.log_generated_interval,
                                         "SHARP LOSS": self.avg_log["avg_sharp_G_loss"] / self.args.log_generated_interval,
                                         "CLS_BAL LOSS": self.avg_log["avg_clsB_G_loss"] / self.args.log_generated_interval,
                                         }, global_step=idx)
                self.avg_log["avg_feat_G_loss"] = 0
                self.avg_log["avg_recon_G_loss"] = 0
                self.avg_log["avg_adv_G_loss"] = 0
                self.avg_log["avg_sharp_G_loss"] = 0
                self.avg_log["avg_clsB_G_loss"] = 0


                fid_score = self.FID.getScore(generated_images_original, real_images_original)
                self.logger.add_scalar("[TRAIN][GEN] FID SCORE", fid_score, global_step=idx)


            # Clip gradients for stabilization
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)
            if self.args.useCond:
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)

            # Take optimizer step
            self.generator_optimizer.step()
            self.mapping_network_optimizer.step()
            if self.args.useCond:
                self.adapter_optimizer.step()

        # Log generated images
        if (idx + 1) % self.args.log_generated_images_interval == 0 or idx==0:
            #tracker.add('generated', torch.cat([generated_images[:6], real_images[:3]], dim=0))
            
            # text = f"Min: {torch.min(real_images_original)}, Max: {torch.max(real_images_original)}"
            # self.logger.add_text("[REAL RGB]", text, global_step=idx)
            # text = f"Min: {torch.min(generated_images_original)}, Max: {torch.max(generated_images_original)}"
            # self.logger.add_text("[FAKE RGB]", text, global_step=idx)

            self.logger.add_scalars("[TRAIN][GEN] RGB RANGE", {"RGB REAL min":torch.min(real_images_original), "RGB REAL MAX":torch.max(real_images_original), "RGB FAKE min": torch.min(generated_images_original), "RGB FAKE MAX": torch.max(generated_images_original)}, global_step=idx)
            
            #mask_visual = torch.argmax(generated_masks.detach(), dim=1).unsqueeze(1) / self.args.num_classes
            #print("mask_visual: ", torch.unique(mask_visual))

            
            colors = generate_distinguishable_colors( self.args.num_classes)
            img_masks = (get_images_with_mask(normalizeRGB(generated_images_original), generated_masks, color=colors) / 255.0).to(self.args.device)
            #mask_visual = mask_visual.repeat(1,3,1,1)
            #print("img mask: ",torch.max(img_masks))
            #print("img original: ",torch.max(generated_images_original))

            gen_visual = torch.cat((normalizeRGB(generated_images_original), img_masks), dim=0)[self.permute_list,:,:,:]
            #print("gen vis: ", gen_visual.shape)
            save_tensorboard_images(gen_visual, "[TRAIN][GEN] SYNTHETIC IMAGES", self.logger, idx, normalize=True, nrow=self.args.n_row*2)
            
            #real_visual = torch.argmax(real_mask.detach(), dim=1).unsqueeze(1)  / self.args.num_classes
            #print("real_visual: ", torch.unique(real_visual))
            #real_visual = real_visual.repeat(1,3,1,1) 
            #print("mask values: ", torch.unique(real_mask[0]))
            img_masks = (get_images_with_mask(real_images_original, real_mask, color=colors) / 255.0).to(self.args.device)

            # matplotlib.use('TkAgg')
            # plt.imshow(real_mask[0].sum(dim=0).detach().cpu().numpy())
            # plt.show()

            real_visual = torch.cat((real_images_original, img_masks), dim=0)[self.permute_list,:,:,:]
            # plt.subplot(2,1,1)
            # plt.imshow(real_visual[0].detach().permute(1,2,0).cpu().numpy())
            # plt.subplot(2,1,2)
            # plt.imshow(real_visual[1].detach().permute(1,2,0).cpu().numpy())
            # plt.show()
            save_tensorboard_images(real_visual, "[TRAIN][GEN] REAL IMAGES", self.logger, idx,  normalize=True, nrow=self.args.n_row*2)
            save_tensorboard_images(real_images, "[TRAIN][GEN][with AUG] REAL IMAGES ", self.logger, idx,  normalize=True, nrow=self.args.n_row)

            # Upsample each tensor to shape Bx3x256x256 using nearest neighbor interpolation
            rescaled_tensors = []
            for tensor in int_state:
                B, C, H, W = tensor[0].shape
                # Upsample using nearest neighbor interpolation
                upsampled_tensor = torch.nn.functional.interpolate(tensor[0], size=(256, 256), mode='nearest')
                rescaled_tensors.append(upsampled_tensor)

            # Concatenate rescaled tensors along the width dimension (dimension 3)
            int_state_vis = torch.cat(rescaled_tensors, dim=3)

            save_tensorboard_images(int_state_vis, "[TRAIN][GEN] INTERNAL STATE ", self.logger, idx,
                                    normalize=True, nrow=self.args.n_row)

            self.logger.add_figure("[~REAL] P(mask,img)", mi2_real["fig"], idx)
            self.logger.add_figure("[~FAKE] P(mask,img)", mi2_fake["fig"], idx)
        
        plt.close(mi2_real["fig"])
        plt.close(mi2_fake["fig"])

        
        # Save model checkpoints
        if (idx + 1) % self.args.save_checkpoint_interval == 0:

            disc_state_rgb = modelCheckpoint(self.discriminator_rgb, self.discriminator_optimizer_rgb, disc_loss_rgb, label="_rgb")
            disc_state_seg = modelCheckpoint(self.discriminator_seg, self.discriminator_optimizer_seg, disc_loss_seg, label="_seg")
            gen_state = modelCheckpoint(self.generator, self.generator_optimizer, gen_loss)
            mapnet_state = modelCheckpoint(self.mapping_network, self.mapping_network_optimizer, gen_loss)
            if self.args.useCond:
                adapter_state = modelCheckpoint(self.adapter, self.adapter_optimizer, gen_loss)

            if self.args.useCond:
                states = [disc_state_rgb, disc_state_seg, gen_state, mapnet_state, adapter_state]
            else:
                states = [disc_state_rgb, disc_state_seg, gen_state, mapnet_state]

            save_states(states, idx, os.path.join(self.args.prefix, self.args.model_path,self.args.train_id), self.p)
            path_name = os.path.join(self.args.prefix, self.args.model_path, self.args.train_id)
            print("Saved state in {} at epoch {}".format(path_name,idx))
            
            #torch.save(self.discriminator.state_dict(), os.path.join(self.args.model_path,self.args.train_id,f"disc_ckpt_i{idx}.pt"))
            #torch.save(self.generator.state_dict(), os.path.join(self.args.model_path,self.args.train_id, f"gen_ckpt_i{idx}.pt"))
            #torch.save(self.mapping_network.state_dict(), os.path.join(self.args.model_path,self.args.train_id, f"mapnet_ckpt_i{idx}.pt"))

        # if self.nan_flag:
        #      print(f"******** NaN detected in disc_loss at step {idx}! ")
            
        #      save_states(self.debug_states, idx-1, os.path.join(self.args.model_path,self.args.train_id), self.p)
        #      print(f"State {idx-1} have been saved in {os.path.join(self.args.model_path,self.args.train_id)}.")
        #      print("Exiting...")
        #      exit()
        # else:
        #     disc_state_rgb = modelCheckpoint(self.discriminator_rgb, self.discriminator_optimizer_rgb, disc_loss_rgb, label="_rgb")
        #     disc_state_seg = modelCheckpoint(self.discriminator_seg, self.discriminator_optimizer_seg, disc_loss_seg, label="_seg")
        #     gen_state = modelCheckpoint(self.generator, self.generator_optimizer, gen_loss)
        #     mapnet_state = modelCheckpoint(self.mapping_network, self.mapping_network_optimizer, gen_loss)

        #     self.debug_states = [disc_state_rgb, disc_state_seg, gen_state, mapnet_state]

        # Flush tracker
        self.logger.flush()

    def train(self):
        """
        ## Train model
        """

        if self.args.pretrain_gen:
            self.pretrainGen()
        elif self.args.ptg_load:
            save_path = os.path.join(self.args.prefix, self.args.model_path, self.args.train_id, f"pretrained_gen.pt")
            checkpoint = torch.load(save_path)
            self.generator.load_state_dict(checkpoint['model_state_dict'])
            self.generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.generator.train()
            print(f"Loaded PRETRAINED GENERATOR from {save_path}")


        # Loop for `training_steps`
        for i in tqdm(range(self.args.training_steps)):
            # Take a training step
            self.step(i)
            #
            if (i + 1) % self.args.log_generated_interval == 0:
                #tracker.new_line()
                pass


def main():
    """
    ### Train StyleGAN2
    """
    #CONFIGURATION
    args = sys.argv
    prefix = args[1] if len(args)>1 else "."
    config_file = args[2] if len(args) > 1 else "config/styleGAN2-ada_dualDrgb_SS.yaml"

    config_file = os.path.join(prefix,config_file)
    print("Loading configuration from ", config_file)
    cfg = Configurator(config_file, prefix)
    args = cfg.getArgs()

    args.categories = ["bg+paper","rigid_plastic", "cardboard", "metal", "soft_plastic"]
    #args.categories = ['skin', 'nose', 'eyes', 'eyebrows', 'ears', 'mouth', 'lip', 'hair', 'hat', 'eyeglass', 'earring', 'necklace', 'neck', 'cloth']

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    save_path = os.path.join(args.prefix, args.model_path,args.train_id,f"used_config.yaml")
    with open(save_path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
        print("Current Configuration saved!")

    #CREATE TRAINER OBJECT
    trainer = Trainer(args)

    trainer.train() #start training loop

    # # Create an experiment
    # experiment.create(name='stylegan2')
    # # Create configurations object
    # configs = Configs()

    # # Set configurations and override some
    # experiment.configs(configs, {
    #     'device.cuda_device': 0,
    #     'image_size': 64,
    #     'log_generated_interval': 200
    # })

    # # Initialize
    # configs.init()
    # # Set models for saving and loading
    # experiment.add_pytorch_models(mapping_network=configs.mapping_network,
    #                               generator=configs.generator,
    #                               discriminator=configs.discriminator)

    # # Start the experiment
    # with experiment.start():
    #     # Run the training loop
    #     configs.train()

if __name__ == '__main__':
    main()
