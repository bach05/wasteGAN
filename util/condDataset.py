import torch.utils.data
from torchvision.utils import make_grid
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from util.customTransforms import *
from torchvision import transforms as T
from PIL import Image
import math
from util.styleGAN import PatchedImageGenerator
import time
import random

class condDataset(Dataset):
    def __init__(self, data_root, patch_root, image_size):
        
        self.dataset_img = dset.ImageFolder(root=data_root,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

        self.dataset_patch = dset.ImageFolder(root=patch_root,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


    def __len__(self):
        return len(self.dataset_img)

    def __getitem__(self, index):

        # print("Range nomralized image = {} - {}".format(np.min(self.dataset_img[index][0].permute(1,2,0).cpu().numpy()), np.max(self.dataset_img[index][0].permute(1,2,0).cpu().numpy())))

        # f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(self.dataset_img[index][0].permute(1,2,0).cpu().numpy())
        # axarr[1].imshow(self.dataset_patch[index][0].permute(1,2,0).cpu().numpy())
        # plt.show()

        return (self.dataset_img[index], self.dataset_patch[index])


class condDatasetLabel(Dataset):
    def __init__(self, data_root, patch_root, label_root, image_size):
        
        self.dataset_img = dset.ImageFolder(root=data_root,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

        self.dataset_patch = dset.ImageFolder(root=patch_root,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
                    
        self.dataset_label = dset.ImageFolder(root=label_root,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               extractChannel(),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

    def __len__(self):
        return len(self.dataset_img)

    def __getitem__(self, index):
        return (self.dataset_img[index], self.dataset_patch[index], self.dataset_label[index])



class customCondDataset(Dataset):
    def __init__(self, data_root, patch_root, image_size=256, transforms=None, size=None):
        
        self.img_path_folder = data_root
        self.patch_path_folder = patch_root
        self.img_size = image_size
        self.transforms = transforms
        self.size = size

        self.img_list = os.listdir(self.img_path_folder)
        #self.patch_list = os.listdir(self.patch_path_folder)

        if self.size is not None: 
            self.img_list= self.img_list[:size]
            #self.patch_list = self.patch_list[:size]


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = cv2.imread(self.img_path_folder+"/"+self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #patch = cv2.imread(self.patch_path_folder+"/"+self.patch_list[idx])
        #patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)


        if self.transforms is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = Image.fromarray(img)
            #patch = Image.fromarray(patch)
            t = T.Compose([T.Resize(self.img_size),T.CenterCrop(self.img_size), T.ToTensor(),  T.Normalize(mean, std)])
            img = t(img)
            #patch = t(patch)
        else:
            input_list = [img] #[img, patch]
            out_list = self.transforms(input_list)
            img = out_list[0]


        return img


class noiseCondDataset(Dataset):
    def __init__(self, data_root, patch_root, image_size, transforms=None, size=None):
        
        self.img_path_folder = data_root
        self.patch_path_folder = patch_root
        self.img_size = image_size
        self.transforms = transforms
        self.size = size
        #self.scheduler_time_constant = scheduler_time_constant
        #self.offset = offset
        #self.timestamp = (self.offset - 1) / self.offset

        self.img_list = os.listdir(self.img_path_folder)
        self.patch_list = os.listdir(self.patch_path_folder)

        if self.size is not None: 
            self.img_list= self.img_list[:size]
            self.patch_list = self.patch_list[:size]


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = cv2.imread(self.img_path_folder+"/"+self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        patch = cv2.imread(self.patch_path_folder+"/"+self.patch_list[idx], cv2.IMREAD_UNCHANGED)
        #print(patch.shape)
        assert patch.shape[2] == 4
        patch = cv2.cvtColor(patch, cv2.COLOR_BGRA2RGBA)

        if self.transforms is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = Image.fromarray(img)
            patch = Image.fromarray(patch)
            t = T.Compose([T.Resize(256),T.CenterCrop(self.img_size), T.ToTensor(),  T.Normalize(mean, std)])
            img = t(img)
            patch = t(patch)
        else:
            mask = patch[:,:,3]
            patch = patch[:,:,:3]

            input_list = [img, patch, mask]
            out_list = self.transforms(input_list)
            img, patch, mask = out_list

        return (img, patch, mask)


class MaskGANDataset(Dataset):
    def __init__(self, data_root, label_root, image_size, n_classes, transforms=None, size=None):
        
        self.img_path_folder = data_root
        self.label_path_folder = label_root
        self.img_size = image_size
        self.transforms = transforms
        self.size = size
        self.n_classes = n_classes
        #self.scheduler_time_constant = scheduler_time_constant
        #self.offset = offset
        #self.timestamp = (self.offset - 1) / self.offset

        self.img_list = os.listdir(self.img_path_folder)
        self.label_list = os.listdir(self.label_path_folder)

        if self.size is not None: 
            self.img_list= self.img_list[:size]
            self.label_list = self.label_list[:size]


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = cv2.imread(self.img_path_folder+"/"+self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = cv2.imread(self.label_path_folder+"/"+self.label_list[idx])
        #print(label.shape)
        #assert patch.shape[2] == 4
        #patch = cv2.cvtColor(patch, cv2.COLOR_BGRA2RGBA)
        label = label[:,:,0] 
        #print("label before: ", np.unique(label))

        if self.transforms is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = Image.fromarray(img)
            label = Image.fromarray(label)
            t = T.Compose([T.Resize(256),T.CenterCrop(self.img_size), T.ToTensor(),  T.Normalize(mean, std, mask=1)])
            img = t(img)
            label = t(label)
        else:
            input_list = [img, label]
            out_list = self.transforms(input_list)
            img, label = out_list
            # fig = plt.figure()
            # plt.subplot(2,1,1)
            # plt.imshow((label*50).permute(1,2,0).detach().cpu().numpy())
            #label = torch.round(label) / self.n_classes #normalize labels
            label = label.repeat(self.n_classes,1,1)
            compare = torch.Tensor(range(self.n_classes))
            compare = compare.unsqueeze(1).unsqueeze(2).repeat(1, img.shape[1], img.shape[2])
            #print("compare: ", compare.shape)
            label = label == compare
            label = label.float()
            
            # grid = make_grid(label.unsqueeze(1), nrow=5)
            # print("label: ", label.unsqueeze(1).shape)
            # print("grid: ", grid.shape)
            # plt.subplot(2,1,2)
            # plt.imshow(grid.detach().permute(1,2,0).cpu().numpy())
            # fig.savefig("/home/alberto/GAN_venv/log.png")
            
            #print("label after: ", torch.unique(label))

        return (img, label)


class MaskGANDatasetWithPatch(Dataset):
    def __init__(self, data_root, label_root, patch_root, image_size, n_classes, transforms=None, size=None):
        
        self.img_path_folder = data_root
        self.label_path_folder = label_root
        self.patch_path_folder = patch_root
        self.img_size = image_size
        self.transforms = transforms
        self.size = size
        self.n_classes = n_classes
        #self.scheduler_time_constant = scheduler_time_constant
        #self.offset = offset
        #self.timestamp = (self.offset - 1) / self.offset

        self.img_list = os.listdir(self.img_path_folder)
        self.label_list = os.listdir(self.label_path_folder)

        self.img_list.sort()
        self.label_list.sort()

        if self.size is not None: 
            self.img_list= self.img_list[:size]
            self.label_list = self.label_list[:size]

        self.patch_list = os.listdir(patch_root)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = cv2.imread(self.img_path_folder+"/"+self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = cv2.imread(self.label_path_folder+"/"+self.label_list[idx])
        
        #print("IMG: ", self.img_path_folder+"/"+self.img_list[idx])
        #print("LABEL: ", self.label_path_folder+"/"+self.label_list[idx])

        #print(label.shape)
        #assert patch.shape[2] == 4
        #patch = cv2.cvtColor(patch, cv2.COLOR_BGRA2RGBA)
        label = label[:,:,0] 
        #print("label before: ", np.unique(label))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.transforms is None:
           
            img = Image.fromarray(img)
            label = Image.fromarray(label)
            t = T.Compose([T.Resize(256),T.CenterCrop(self.img_size), T.ToTensor(),  T.Normalize(mean, std, mask=1)])
            img = t(img)
            label = t(label)
        else:
            input_list = [img, label]
            out_list = self.transforms(input_list)
            img, label = out_list
            # fig = plt.figure()
            # plt.subplot(2,1,1)
            # plt.imshow((label*50).permute(1,2,0).detach().cpu().numpy())
            #label = torch.round(label) / self.n_classes #normalize labels
            label = label.repeat(self.n_classes,1,1)
            compare = torch.Tensor(range(self.n_classes))
            compare = compare.unsqueeze(1).unsqueeze(2).repeat(1, img.shape[1], img.shape[2])
            #print("compare: ", compare.shape)
            label = label == compare
            label = label.float()
            
            # grid = make_grid(label.unsqueeze(1), nrow=5)
            # print("label: ", label.unsqueeze(1).shape)
            # print("grid: ", grid.shape)
            # plt.subplot(2,1,2)
            # plt.imshow(grid.detach().permute(1,2,0).cpu().numpy())
            # fig.savefig("/home/alberto/GAN_venv/log.png")
            
            #print("label after: ", torch.unique(label))

        patch_name = random.choice(self.patch_list)
        #print(patch_name)
        patch_img = cv2.imread(os.path.join(self.patch_path_folder, patch_name))
        patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)
        ratio = 1200 // self.img_size
        res = 1200 // ratio
        t = T.Compose([T.Resize(res), T.RandomCrop(self.img_size), T.ToTensor(),  T.Normalize(mean, std)])
        patch_img = t(Image.fromarray(patch_img))


        return (img, label, patch_img)


class PatchDataset(Dataset):
    def __init__(self, patch_root, image_size):
        
        self.patch_path_folder = patch_root
        self.img_size = image_size

        self.patch_list = os.listdir(patch_root)

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        patch_name = random.choice(self.patch_list)
        patch_img = cv2.imread(os.path.join(self.patch_path_folder, patch_name))
        patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)
        ratio = 1200 // self.img_size
        res = 1200 // ratio
        t = T.Compose([T.Resize(res), T.RandomCrop(self.img_size), T.ToTensor(),  T.Normalize(mean, std)])
        patch_img = t(Image.fromarray(patch_img))

        return patch_img


class noiseCondDataset_test(Dataset):
    def __init__(self, patch_root, image_size, transforms=None):
        
        self.patch_path_folder = patch_root
        self.img_size = image_size
        self.transforms = transforms
        #self.scheduler_time_constant = scheduler_time_constant
        #self.offset = offset
        #self.timestamp = (self.offset - 1) / self.offset

        self.patch_list = os.listdir(self.patch_path_folder)


    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):

        #print(self.patch_path_folder+"/"+self.patch_list[idx])
        patch = cv2.imread(self.patch_path_folder+"/"+self.patch_list[idx], cv2.IMREAD_UNCHANGED)
        #print(patch.shape)
        assert patch.shape[2] == 4
        patch = cv2.cvtColor(patch, cv2.COLOR_BGRA2RGBA)

        if self.transforms is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            patch = Image.fromarray(patch)
            t = T.Compose([T.Resize(256),T.CenterCrop(self.img_size), T.ToTensor(),  T.Normalize(mean, std)])
            patch = t(patch)
        else:
            mask = patch[:,:,3]
            patch = patch[:,:,:3]

            input_list = [patch, mask]
            out_list = self.transforms(input_list)
            patch, mask = out_list

        return (patch, mask)


class DiffusionCondDataset(Dataset):
    def __init__(self, data_root, patch_root, transforms=None, size=None):
        
        self.img_path_folder = data_root
        self.patch_path_folder = patch_root
        self.transforms = transforms
        self.size = size
        #self.scheduler_time_constant = scheduler_time_constant
        #self.offset = offset
        #self.timestamp = (self.offset - 1) / self.offset

        self.img_list = os.listdir(self.img_path_folder)
        self.patch_list = os.listdir(self.patch_path_folder)

        if self.size is not None: 
            self.img_list= self.img_list[:size]
            self.patch_list = self.patch_list[:size]


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = cv2.imread(self.img_path_folder+"/"+self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        patch = cv2.imread(self.patch_path_folder+"/"+self.patch_list[idx], cv2.IMREAD_UNCHANGED)
        #print(patch.shape)
        assert patch.shape[2] == 4
        patch = cv2.cvtColor(patch, cv2.COLOR_BGRA2RGBA)

        if self.transforms is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = Image.fromarray(img)
            patch = Image.fromarray(patch)
            t = T.Compose([T.Resize(256),T.CenterCrop(self.img_size), T.ToTensor(),  T.Normalize(mean, std)])
            img = t(img)
            patch = t(patch)
        else:
            mask = patch[:,:,3]
            patch = patch[:,:,:3]

            input_list = [img, patch, mask]
            out_list = self.transforms(input_list)
            img, patch, mask = out_list

        return (img, patch, mask)


class SegmentationDataset(Dataset):
    def __init__(self, data_root, mask_root, transforms=None, size=None):
        
        self.img_path_folder = data_root
        self.mask_folder = mask_root
        self.transforms = transforms
        self.size = size
        #self.scheduler_time_constant = scheduler_time_constant
        #self.offset = offset
        #self.timestamp = (self.offset - 1) / self.offset

        self.img_list = os.listdir(self.img_path_folder)
        self.mask_list = os.listdir(self.mask_folder)

        if self.size is not None: 
            self.img_list= self.img_list[:size]
            self.mask_list = self.mask_list[:size]


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = cv2.imread(self.img_path_folder+"/"+self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_folder+"/"+self.mask_list[idx])
        #print(patch.shape)
        mask = mask[:,:,0].astype(np.uint8)
        
        #print("mask: ",np.unique(mask))

        if self.transforms is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = Image.fromarray(img)
            patch = Image.fromarray(patch)
            t = T.Compose([T.Resize(256),T.CenterCrop(self.img_size), T.ToTensor(),  T.Normalize(mean, std)])
            img = t(img)
            patch = t(patch)
        else:

            input_list = [img, mask]
            out_list = self.transforms(input_list)
            img, mask = out_list
            #print("mask: ",mask.shape)

        return (img, mask)







       