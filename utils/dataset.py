from torch.utils.data import Dataset
import cv2
from PIL import Image
import os
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np
import random

class SemSegDataset(Dataset):

    def __init__(self, img_path_folder, mask_path_folder, transform=None, patch=False, mode="train"):
        
        self.img_path_folder = img_path_folder
        self.mask_path_folder = mask_path_folder
        self.mode = mode
        self.transform = transform
        self.patches = patch

        self.img_list = os.listdir(self.img_path_folder)
        self.mask_list = os.listdir(self.mask_path_folder)

        self.img_list.sort()
        self.mask_list.sort()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = cv2.imread(self.img_path_folder+"/"+self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path_folder+"/"+self.mask_list[idx], cv2.IMREAD_GRAYSCALE)

        #print(f"index img {idx}/{len(self.img_list)}")
        #print(f"index mask {idx}/{len(self.mask_list)}")
        
        # if  self.img_list[idx] != self.mask_list[idx]:
        #     print("WARNING!!! ")
        #     print("IMG:", self.img_list[idx] )
        #     print("MASK:", self.mask_list[idx] )

        # plt.subplot(2,1,1)
        # plt.imshow(mask)
        # plt.subplot(2,1,2)
        # plt.imshow(img)
        # plt.show()

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img =aug['image']
            mask = aug['mask']
            # plt.subplot(2,2,3)
            # plt.imshow(mask)
            # plt.subplot(2,2,4)
            # plt.imshow(img)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            img = t(img)
            t = T.Compose([T.ToTensor()])
            mask = t(mask.astype(float)).squeeze().long()

        #plt.show()

        #print("mask: ", np.unique(mask))


        if self.transform is None:

            img = Image.fromarray(img)
            mask = Image.fromarray(mask)
            #print("mask pil: ",np.unique((np.asarray(mask))))
        #   img = Image.fromarray(img)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            t = T.Compose([T.Resize((1430, 1920)), T.CenterCrop(1430), T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean, std)])
            #t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            img = t(img)
            t = T.Compose([T.Resize((1430, 1920), interpolation=T.InterpolationMode.NEAREST), T.CenterCrop(1430), T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])
            #t = T.Compose([T.ToTensor()])
            mask = t(mask).squeeze()
            mask = ((mask*255).round()).long()
            #print("mask: ", torch.unique(mask))
            #print("mask: ", np.unique(mask))

        #print("SS out img: ", img.shape)

        return img, mask


class FilteredSemSegDataset(Dataset):

    def __init__(self, img_path_folder, mask_path_folder, transform=None, patch=False, mode="train", allowed_labels = None):
        
        self.img_path_folder = img_path_folder
        self.mask_path_folder = mask_path_folder
        self.mode = mode
        self.transform = transform
        self.patches = patch

        self.img_list = os.listdir(self.img_path_folder)
        self.mask_list = os.listdir(self.mask_path_folder)

        self.allowed_labels = allowed_labels

        self.img_list.sort()
        self.mask_list.sort()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = cv2.imread(self.img_path_folder+"/"+self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path_folder+"/"+self.mask_list[idx], cv2.IMREAD_GRAYSCALE)

        if self.allowed_labels is not None:
             for label in self.allowed_labels:
                filter = mask == label
                filter_not = mask != label
                mask[filter] = 1
                mask[filter_not] = 0

        #print(f"index img {idx}/{len(self.img_list)}")
        #print(f"index mask {idx}/{len(self.mask_list)}")
        
        if  self.img_list[idx] != self.mask_list[idx]:
            print("WARNING!!! ")
            print("IMG:", self.img_list[idx] )
            print("MASK:", self.mask_list[idx] )

        # plt.subplot(2,1,1)
        # plt.imshow(mask)
        # plt.subplot(2,1,2)
        # plt.imshow(img)
        # plt.show()

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img =aug['image']
            mask = aug['mask']
            # plt.subplot(2,2,3)
            # plt.imshow(mask)
            # plt.subplot(2,2,4)
            # plt.imshow(img)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            img = t(img)
            t = T.Compose([T.ToTensor()])
            mask = t(mask.astype(float)).squeeze().long()

        #plt.show()

        #print("mask: ", np.unique(mask))


        if self.transform is None:
            img = Image.fromarray(img)
            mask = Image.fromarray(mask)
            #print("mask pil: ",np.unique((np.asarray(mask))))
        #   img = Image.fromarray(img)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            t = T.Compose([T.Resize((1430, 1920)), T.CenterCrop(1430), T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean, std)])
            #t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            img = t(img)
            t = T.Compose([T.Resize((1430, 1920)), T.CenterCrop(1430), T.Resize((256, 256)), T.ToTensor()])
            #t = T.Compose([T.ToTensor()])
            mask = t(mask).squeeze()
            mask = (mask>0).long()
            #print("mask: ", torch.unique(mask))
            #print("mask: ", np.unique(mask))

        #print("SS out img: ", img.shape)
        #print("Image: ", img.shape)
        #print("Mask: ", mask.shape)

        return img, mask


class AugSemSegDataset(Dataset):

    def __init__(self, img_path_folder, mask_path_folder, aug_folder, aug_p=0.25, transform=None, patch=False, mode="train", real_size=None):
        
        self.img_path_folder = img_path_folder
        self.mask_path_folder = mask_path_folder
        self.aug_folder_img = os.path.join(aug_folder, "images")
        self.aug_folder_label = os.path.join(aug_folder, "labels")
        self.mode = mode
        self.transform = transform
        self.patches = patch

        self.img_list = os.listdir(self.img_path_folder)
        self.mask_list = os.listdir(self.mask_path_folder)
        self.aug_img_list = os.listdir(self.aug_folder_img)
        self.aug_label_list = os.listdir(self.aug_folder_label)

        self.img_list.sort()
        self.mask_list.sort()
        self.aug_label_list.sort()
        self.aug_img_list.sort()

        #print("aug mist: ", len(self.aug_img_list))
        random.seed(11)
        if real_size is not None:
            c = list(zip(self.img_list, self.mask_list))
            c = random.sample(c, real_size)
            self.img_list, self.mask_list = zip(*c)
            self.img_list = list(self.img_list)
            self.mask_list = list(self.mask_list)

        print("REAL TRAINING SAMPLES: ", len(self.img_list))
        #print("REAL SIZE: ", real_size)

        m = int(len(self.img_list)*aug_p)
        if m > 0:

            random_list = random.sample(list(zip(self.aug_img_list, self.aug_label_list)), m)
            aug_img_sub, aug_label_sub = zip(*random_list)
     
            print("FAKE TRAINING SAMPLES: ", len(aug_img_sub))

            self.img_list = self.img_list+list(aug_img_sub)
            self.mask_list = self.mask_list+list(aug_label_sub)

        print("TOTAL TRAINING SAMPLES: ", len(self.img_list))
        


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        #print("IMG:", self.img_list[idx] )
        #print("MASK:", self.mask_list[idx] )
        if self.img_list[idx][:6] == "augImg":
            img = cv2.imread(os.path.join(self.aug_folder_img, self.img_list[idx]))
            mask =  cv2.imread(os.path.join(self.aug_folder_label, self.mask_list[idx]), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(self.img_path_folder+"/"+self.img_list[idx])
            mask = cv2.imread(self.mask_path_folder+"/"+self.mask_list[idx], cv2.IMREAD_GRAYSCALE)
      
        # if img is None:
        #     os.path.join(self.aug_folder_img, self.img_list[idx])
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # plt.subplot(2,2,1)
        # plt.imshow(mask)
        # plt.subplot(2,2,2)
        # plt.imshow(img)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img =aug['image']
            mask = aug['mask']
            # plt.subplot(2,2,3)
            # plt.imshow(mask)
            # plt.subplot(2,2,4)
            # plt.imshow(img)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            img = t(img)
            t = T.Compose([T.ToTensor()])
            mask = t(mask.astype(float)).squeeze().long()

        #plt.show()

        #print("mask: ", np.unique(mask))


        if self.transform is None:
            img = Image.fromarray(img)
            mask = Image.fromarray(mask)
            #print("mask pil: ",np.unique((np.asarray(mask))))
        #   img = Image.fromarray(img)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            if self.img_list[idx][:6] == "augImg":
                t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
                img = t(img)
                t = T.Compose([T.ToTensor()])
                mask = t(mask).squeeze()
                mask = ((mask*255).round()).long()
                #print("AUG mask: ", torch.unique(mask))

            else:
                t = T.Compose([T.Resize((1430, 1920)), T.CenterCrop(1430), T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean, std)])
                img = t(img)
                t = T.Compose([T.Resize((1430, 1920),interpolation=T.InterpolationMode.NEAREST), T.CenterCrop(1430), T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])
                mask = t(mask).squeeze()
                mask = ((mask*255).round()).long()
                #print("GT mask: ", torch.unique(mask))

            #print("mask: ", torch.unique(mask))
            #print("mask: ", np.unique(mask))

        #print(img.shape)
        #print("ASS out img: ", img.shape)


        return img, mask


class InferenceSemSegDataset(Dataset):

    def __init__(self, img_path_folder, transform=None, patch=False, mode="train"):
        
        self.img_path_folder = img_path_folder
        self.mode = mode
        self.transform = transform
        self.patches = patch

        self.img_list = os.listdir(self.img_path_folder)

        self.img_list.sort()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = cv2.imread(self.img_path_folder+"/"+self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #print("IMG:", self.img_list[idx] )
        #print("MASK:", self.mask_list[idx] )

        # plt.subplot(2,2,1)
        # plt.imshow(mask)
        # plt.subplot(2,2,2)
        # plt.imshow(img)

        if self.transform is not None:
            aug = self.transform(image=img)
            img =aug['image']
            # plt.subplot(2,2,3)
            # plt.imshow(mask)
            # plt.subplot(2,2,4)
            # plt.imshow(img)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            img = t(img)

        #plt.show()

        if self.transform is None:
            img = Image.fromarray(img)
        #   img = Image.fromarray(img)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            t = T.Compose([T.Resize((1430, 1920)), T.CenterCrop(1430), T.Resize((608, 608)), T.ToTensor(), T.Normalize(mean, std)])
            img = t(img)


        #print(img.shape)

        return img

