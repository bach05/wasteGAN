import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torchvision.utils import draw_segmentation_masks
import colorsys

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
    
    #print("mask logits: ", masks_logits.shape)
    
    B, C, H, W = masks_logits.shape
    
    images = normalizeRGB(images.detach(), use_int8=True).to(device)
    normalized_masks = torch.nn.functional.softmax(masks_logits.detach(), dim=1)
    mask_idx =  torch.argmax(masks_logits.detach(), dim=1).unsqueeze(1)
    #print("mask_idx", mask_idx.shape)
    masks = torch.zeros_like(masks_logits).to(torch.bool).to(device)
    for i in range(C):
        masks[:,i,:,:] = mask_idx[:,0,:,:] == i

    #print("masks", masks.shape)
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


def generate_distinguishable_colors(k):
    colors = []
    for i in range(k):
        hue = i / k  # Vary the hue component
        saturation = 0.7  # You can adjust this to control saturation
        lightness = 0.6  # You can adjust this to control lightness
        rgb_color = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = "#{:02X}{:02X}{:02X}".format(
            int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255)
        )
        colors.append(hex_color)
    return colors
