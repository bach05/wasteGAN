import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from util.condDataset import condDataset, customCondDataset, DiffusionCondDataset
from util.customTransforms import *
from torchvision.utils import draw_segmentation_masks


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def save_tensorboard_images(images, label, logger, iters, **kwargs):
    grid = torchvision.utils.make_grid(images,**kwargs)
    logger.add_image(label, grid, iters)

def plot_grid_images(images, **kwargs):
    grid = torchvision.utils.make_grid(images,**kwargs)
    plt.imshow(grid.detach().permute(1,2,0).cpu().numpy())

def get_images_with_mask(images, masks_logits, color=None, device="cpu", alpha=0.5, **kwargs):
    B, C, H, W = masks_logits.shape
    
    images = normalizeRGB(images.detach(), use_int8=True).to(device)
    normalized_masks = torch.nn.functional.softmax(masks_logits.detach(), dim=1)
    mask_idx =  torch.argmax(normalized_masks, dim=1).unsqueeze(1)
    masks = torch.zeros_like(masks_logits).to(torch.bool).to(device)
    for i in range(C):
        masks[:,i,:,:] = mask_idx[:,0,:,:] == i

    img_with_masks = [
    draw_segmentation_masks(img, masks=mask, alpha=alpha, colors=color).unsqueeze(0)
    for img, mask in zip(images, masks)
    ]

    return torch.cat(img_with_masks, 0)

def normalizeRGB(images, use_int8=False):

        B, C, H, W = images.shape
        max = torch.max(images.view(B,C,H*W), dim=2)[0]
        min = torch.min(images.view(B,C,H*W), dim=2)[0]
        #print("max: ", max.shape)

        max = max.unsqueeze(2).unsqueeze(3).repeat(1,1,H,W)
        min = min.unsqueeze(2).unsqueeze(3).repeat(1,1,H,W)
        #print("max: ", max.shape)

        images = (images - min) / (max -min)

        if use_int8:
            images = (images * 255).to(torch.uint8)
        
        return images

    



def get_data(args):
    dataroot = args.dataroot
    patchroot = args.patchroot
    image_size = args.image_size
    batch_size = args.batch_size
    workers = 4
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transforms = MultiCompose([MultiResize(512), MultiRandomCrop(image_size), MultiRandomVerticalFlip(), MultiHorizontalFlip(), MultiToTensor(), MultiNormalize(mean, std)])
    dataset = DiffusionCondDataset(dataroot, patchroot, transforms=transforms)

    if args.debug:
        indices = torch.arange(50)
        dataset = data_utils.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

    return dataloader

def get_val_data(args):
    dataroot = args.dataroot_val
    patchroot = args.patchroot_val
    image_size = args.image_size
    val_size = args.val_batch_size
    workers = 2
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transforms = MultiCompose([MultiResize(512), MultiRandomCrop(256, mode='center'), MultiToTensor(), MultiNormalize(mean, std)])
    dataset = DiffusionCondDataset(dataroot, patchroot, transforms=transforms)

    indices = torch.arange(val_size)
    dataset = data_utils.Subset(dataset, indices)
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=val_size,
                                         shuffle=True, num_workers=workers)

  

    
    return dataloader

   


def setup_logging(run_name):
    os.makedirs("DM_models", exist_ok=True)
    os.makedirs("DM_results", exist_ok=True)
    os.makedirs(os.path.join("DM_models", run_name), exist_ok=True)
    os.makedirs(os.path.join("DM_results", run_name), exist_ok=True)