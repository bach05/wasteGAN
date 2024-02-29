import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms.functional as TF
import math
from torch import autograd
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt


from random import seed
from random import randint
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import time

def cycle_dataloader(data_loader):
    """
    <a id="cycle_dataloader"></a>

    ## Cycle Data Loader

    Infinite loader that recycles the data loader after each epoch
    """
    while True:
        for batch in data_loader:
            yield batch


class pSchedulerCustom(object):

    def __init__(self, target_acc=0.7, init_p=0, integration_steps=1000):
        self.cumulative_outputD = 0
        self.cont = 0
        self.target_acc = target_acc
        self.p = init_p
        self.integration_steps = integration_steps
        self.realD_avg = 0
        self.fakeD_avg = 0

    def step(self, value, value2):

        #self.cumulative_outputD += np.mean(0.5 * (1 + np.sign(value.detach().cpu().numpy()))) #old
        
        self.cumulative_outputD += np.mean(value.detach().cpu().numpy()) #new
        
        self.realD_avg += value
        self.fakeD_avg += value2
        self.cont+=1
        #print("step CoD: ", self.cumulative_outputD)

    def get_p(self, reset=True):
        
        avg_current_acc = 0
        avg_diff = 0
        if self.cumulative_outputD!=0:

            #coD = np.array(self.cumulative_outputD) #old
            #avg_current_acc = np.mean(0.5 * (1 + np.sign(coD))) #old
            
            avg_current_acc = self.cumulative_outputD / self.cont #new
            
            avg_error = avg_current_acc - self.target_acc
            avg_diff = abs(np.mean( (self.realD_avg.detach().cpu().numpy() / self.cont) - (self.fakeD_avg.detach().cpu().numpy() / self.cont) ))

            self.p = np.clip(self.p + (avg_error / self.integration_steps)*avg_diff, 0, 0.75)
            
            #print("COD: ", coD, "avg_curr_acc: ", avg_current_acc, "avg_curr_err: ", avg_error, "p: ", self.p)

        if reset: 
            self.cumulative_outputD = 0
            self.realD_avg = 0
            self.fakeD_avg = 0
            self.cont = 0
        
        return self.p, avg_current_acc, avg_diff
    
class pScheduler(object):

    def __init__(self, target_acc=0.7, init_p=0, integration_steps=1000):
        self.cumulative_outputD = 0
        self.cont = 0
        self.target_acc = target_acc
        self.p = init_p
        self.integration_steps = integration_steps
        self.realD_avg = 0
        self.fakeD_avg = 0

    def step(self, value, value2):

        self.cumulative_outputD += np.mean(0.5 * (1 + np.sign(value.detach().cpu().numpy()))) #old
        
        self.realD_avg += value
        self.fakeD_avg += value2
        self.cont+=1
        #print("step CoD: ", self.cumulative_outputD)

    def get_p(self, reset=True):
        
        avg_current_acc = 0
        avg_diff = 0
        if self.cumulative_outputD!=0:

            coD = np.array(self.cumulative_outputD) #old
            avg_current_acc = np.mean(0.5 * (1 + np.sign(coD))) #old
            
            avg_error = avg_current_acc - self.target_acc
            avg_diff = abs(np.mean( (self.realD_avg.detach().cpu().numpy() / self.cont) - (self.fakeD_avg.detach().cpu().numpy() / self.cont) ))

            self.p = np.clip(self.p + (avg_error / self.integration_steps)*avg_diff, 0, 0.75)
            
            #print("COD: ", coD, "avg_curr_acc: ", avg_current_acc, "avg_curr_err: ", avg_error, "p: ", self.p)

        if reset: 
            self.cumulative_outputD = 0
            self.realD_avg = 0
            self.fakeD_avg = 0
            self.cont = 0
        
        return self.p, avg_current_acc, avg_diff

class modelCheckpoint(object):
    def __init__(self, model, optimizer, loss, label=""):
        self.model = model.state_dict()
        self.optimizer = optimizer.state_dict()
        self.loss = loss
        self.name = model.__class__.__name__ + label
    def getState(self):
        return self.model, self.optimizer, self.loss
    def getName(self):
        return self.name

def save_checkpoint(model, optimizer, loss, epoch, save_path, p=None):
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model,
            'optimizer_state_dict': optimizer,
            'loss': loss,
            'p': p,
            }, save_path)

#save a lit of states
def save_states(states: modelCheckpoint, epoch, save_path, p=None):
    
    current_subfolder = os.path.join(save_path, f"step_{epoch}")
    if not os.path.exists(current_subfolder):
        os.makedirs(current_subfolder)

    for state in states:
        model, optimizer, loss = state.getState()
        name = state.getName()
        save_checkpoint(model, optimizer, loss, epoch, os.path.join(current_subfolder, f"{name}_ckpt.pt"), p)

def debug_NaN(label, image, logger, step, file):
    if torch.any(image.isnan()):
        print(f"{label} images are NaN")
        file.debug(f"{label} images are NaN")
        return True
    else:
        #logger.add_scalar(f"|DEBUG NAN| AVG {label} IMG: ", torch.mean(image), global_step=step)
        return False

#create a pyramind of features
class ConditionalFeatures(nn.Module):
    def __init__(self, init_res, num_feat):

        super(ConditionalFeatures, self).__init__()
        self.n_feat = num_feat
        self.res = init_res
        
    def forward(self, input):
        
        bw_input = TF.rgb_to_grayscale(input)

        pyramid = [bw_input]
        for i in range(self.n_feat-1):
            self.res = self.res // 2
            pyramid.append(TF.resize(img=bw_input, size=self.res, antialias=True))

        return pyramid


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

class FIDscore(nn.Module):

    def __init__(self, features=64, device="cpu"):
        super().__init__()
        self.fid = FrechetInceptionDistance(feature=features, normalize=True).to(device)

    def getScore(self, fake_img, real_image):

        self.fid.update(real_image, real=True)
        self.fid.update(fake_img, real=False)
        score = self.fid.compute()
        self.fid.reset()

        return score #lower is better


class PatchedImageGenerator():
    def __init__(self, data_root, obj_count=20, resolution=256, pos_ratio=0.1):
        super().__init__()
        self.data_root = data_root

         # seed random number generator
        self.rseed = seed(time.time)
        #print("RANDOM SEED: ",rseed)

        #random params
        self.x_res = 1200
        self.y_res = 1944
        rand_pos_x = randint(0,self.x_res)
        rand_pos_y = randint(0,self.y_res)
        rand_rot =  0
        self.out_res = resolution

        #load lists

        ### TETRAPAK
        root = os.path.join(self.data_root, "dataset/tetrapack/")
        rgb = next(os.walk(root + "rgb"), (None, None, []))[2]
        labels = next(os.walk(root + "mask"), (None, None, []))[2]

        self.tetrapak_img = [] #empty list, index 0 = rgb images, index 2 = labels
        for path_rgb, path_label in zip(rgb,labels):
            #img = Image.open(root+"rgb/"+path_rgb)
            #label = Image.open(root+"mask/"+path_label)
            path_rgb = root +"rgb/" + path_rgb 
            path_label = root +"mask/" + path_label
            self.tetrapak_img.append([path_rgb, path_label])


        ### PAPER
        root = os.path.join(self.data_root, "dataset/paper/")
        rgb = next(os.walk(root + "rgb"), (None, None, []))[2]
        labels = next(os.walk(root + "mask"), (None, None, []))[2]

        self.paper_img = [] #empty list, index 0 = rgb images, index 2 = labels
        for path_rgb, path_label in zip(rgb,labels):
            #img = Image.open(root+"rgb/"+path_rgb)
            #label = Image.open(root+"mask/"+path_label)
            path_rgb = root +"rgb/" + path_rgb 
            path_label = root +"mask/" + path_label
            self.paper_img.append([path_rgb, path_label])

        ### CARDBOARD
        root = os.path.join(self.data_root, "dataset/cardboard/")
        rgb = next(os.walk(root + "rgb"), (None, None, []))[2]
        labels = next(os.walk(root + "mask"), (None, None, []))[2]

        self.cardboard_img = [] #empty list, index 0 = rgb images, index 2 = labels
        for path_rgb, path_label in zip(rgb,labels):
            #img = Image.open(root+"rgb/"+path_rgb)
            #label = Image.open(root+"mask/"+path_label)
            path_rgb = root +"rgb/" + path_rgb 
            path_label = root +"mask/" + path_label
            self.cardboard_img.append([path_rgb, path_label])


        ### OTHER
        root = os.path.join(self.data_root, "dataset/other/")
        rgb = next(os.walk(root + "rgb"), (None, None, []))[2]
        labels = next(os.walk(root + "mask"), (None, None, []))[2]

        self.other_img = [] #empty list, index 0 = rgb images, index 2 = labels
        for path_rgb, path_label in zip(rgb,labels):
            #img = Image.open(root+"rgb/"+path_rgb)
            #label = Image.open(root+"mask/"+path_label)
            path_rgb = root +"rgb/" + path_rgb 
            path_label = root +"mask/" + path_label 
            self.other_img.append([path_rgb, path_label])

        self.count_tetrapak = len(self.tetrapak_img)
        count_paper = len(self.paper_img)
        count_cardboard = len(self.cardboard_img)
        count_other = len(self.other_img)

        # print("LOADED DATA: ")
        # print("TETRAPACK: ",len(self.tetrapak_img)," items")
        # print("PAPER: ",len(self.paper_img)," items")
        # print("CARDBOARD: ",len(self.cardboard_img)," items")
        # print("OTHER: ",len(self.other_img)," items")

        ## FUSE LISTS OF NON TETRAPACK
        self.non_tetrapak_img = self.paper_img + self.cardboard_img + self.other_img
        self.count_non_tetrapak = len(self.non_tetrapak_img)

        ### GENERATES CLUTTERED IMAGES 
        self.pos_example_count = int(obj_count * pos_ratio)
        self.neg_example_count = obj_count - self.pos_example_count

    
    def getPatchedIamge(self):

        #Load background
        conveyor_belt = Image.open(os.path.join(self.data_root,"bg","belt4.jpeg"))
        
        #background enhancing
        area = (0, 0, self.y_res, self.x_res)
        conveyor_belt = conveyor_belt.crop(area)
        conveyor_belt = conveyor_belt.filter(ImageFilter.GaussianBlur(radius = 15))
        conveyor_belt = ImageOps.flip(conveyor_belt)
        enhancer = ImageEnhance.Brightness(conveyor_belt) #image brightness enhancer
        factor = 1.5 #gives original image
        conveyor_belt = enhancer.enhance(factor)

        conveyor_belt = np.array(conveyor_belt)

        indexes = np.random.RandomState(self.rseed).permutation(self.count_tetrapak)
        tetrapak_idx = indexes[:self.pos_example_count]
        indexes = np.random.RandomState(self.rseed).permutation(self.count_non_tetrapak)
        non_tetrapak_idx = indexes[:self.neg_example_count]

        obj_list = [self.tetrapak_img[index] for index in  tetrapak_idx] + [self.non_tetrapak_img[index] for index in non_tetrapak_idx]
        #print("obj list: ", self.tetrapak_img)
        
        indexes = np.random.RandomState(self.rseed).permutation(len(obj_list))
        
        alpha = np.ones((self.x_res,self.y_res,1),np.uint8) + 255
        #print("conveyor_belt: ", conveyor_belt.shape)
        #print("alpha: ", alpha.shape)
        out_img = Image.fromarray(np.concatenate((conveyor_belt, alpha),axis=2))
        out_mask = Image.fromarray(np.zeros((self.x_res, self.y_res), np.uint8))
        
        for j in indexes:
            out_img, out_mask =  self.randomPatchImg(obj_list[j][0], obj_list[j][1], out_img, out_mask)

        ratio = self.x_res // self.out_res
        #print("ratio: ", ratio)

        out_img = np.array(TF.center_crop(TF.resize(out_img, self.x_res // ratio), self.out_res) )
        
        #plt.imshow(out_img[:,:,:3])
        #plt.show()
        
        return out_img, out_mask


    def randomPatchImg(self, fg_img_path, fg_mask_path, bg_img, bg_label):

        #print(fg_mask_path)
        #print(fg_img_path)
        fg_img = Image.open(fg_img_path)
        fg_mask = Image.open(fg_mask_path)
        assert fg_img.size[:2] == fg_mask.size[:2] and bg_img.size[:2] == bg_label.size[:2]
        x_res = fg_img.size[0]-100
        y_res = fg_img.size[1]-100
        x_offset = randint(100, x_res)
        y_offset = randint(100, y_res)
        angle_offset = randint(0, 360)
        
        #RGB IMAGE
        fg_img = fg_img.crop(fg_img.getbbox())
        fg_img = fg_img.rotate(angle_offset, resample=Image.BICUBIC, expand=True)
        #print("bg_img: ", bg_img.mode)
        #print("fg_img: ", fg_img.mode)
        Image.Image.paste(bg_img, fg_img, (x_offset, y_offset), fg_img)

        #LABEL
        fg_mask = self.remapLabel(fg_mask, 50, 100)
        fg_mask = fg_mask.crop(fg_mask.getbbox())
        fg_mask = fg_mask.rotate(angle_offset, resample=Image.BICUBIC, expand=True)
        #print("bg_label: ", bg_label.mode)
        #print("fg_mask: ", fg_mask.mode)
        #print("fg_img: ", fg_img.mode)
        Image.Image.paste(bg_label, fg_mask, (x_offset, y_offset), fg_img)

        #bg_img.show()
        #bg_label.show()
        #input("Continue...")

        #y1, y2 = y_offset, y_offset + img.shape[0]
        #x1, x2 = x_offset, x_offset + img.shape[1]

        return bg_img, bg_label

    def remapLabel(self, img,  pos_label, neg_label):
        
        imgP = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if imgP[i,j] == 246:
                    imgP[i,j] = pos_label
                elif imgP[i,j] != 0:
                    imgP[i,j] = neg_label
        
        return img
        


