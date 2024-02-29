import torch
import numpy as np
import random
import cv2
import collections

class extractChannel(object):
    """Convert extract a single channel from an image"""

    def __call__(self, image):

        red, green, blue = image.split()
        return red

class MultiCompose(object):
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, input_list):

        #print("[COMP] in list: ",len(input_list))

        for t in self.segtransform:
            input_list = t(input_list)
        return input_list

class MultiToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, input_list):

        #print("[TT] in list: ",len(input_list))
        out_list = []
        for image in input_list:
            if not isinstance(image, np.ndarray):
                raise (RuntimeError("MultiToTensor() only handle np.ndarray"
                                    "[eg: data readed by cv2.imread()].\n"))
            if len(image.shape) > 3 or len(image.shape) < 2:
                raise (RuntimeError("MultiToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)

            image = torch.from_numpy(image.transpose((2, 0, 1)))
    
            if not isinstance(image, torch.FloatTensor):
                image = image.float()
            
            out_list.append(image)

        #print("out list: ",out_list)
        return out_list 

class MultiNormalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None, mask=None): #mask=mask index in the list
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std
        self.mask = mask

    def __call__(self, input_list):
        
        out_list = []
        #print("[N] in list: ",len(input_list))

        for i,image in enumerate(input_list):
            
            if self.mask==i:
                out_list.append(image)
                continue

            #print("image 1: {} - {}".format(torch.min(image), torch.max(image)))
            # for t in image:
            #     if torch.max(t) != 0:
            #         t = t.div_(torch.max(t))
            #image = image.div_(torch.max(torch.max(image, dim=1).values, dim=1).values)
            #print("image: {} - {}".format(torch.min(image), torch.max(image)))
            
            image = image.to(torch.float32)/255.0

            #print("image2: {} - {}".format(torch.min(image), torch.max(image)))
            
            if self.std is None:
                for t, m in zip(image, self.mean):
                    #print(type(t))
                    t.sub_(m)
            else:
                for t, m, s in zip(image, self.mean, self.std):
                    #print(type(t))
                    t.sub_(m).div_(s)
            
            #print("norm image: {} - {}".format(torch.min(image), torch.max(image)))
            out_list.append(image)
        
        return out_list

class MultiResize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, size):
        self.size = size

    def __call__(self, input_list):
       
        #print("[R] in list: ",len(input_list))
        
        h,w,c = input_list[0].shape
        assert self.size <= w, f"Error: {h}x{w}x{c}"
        ratio = self.size / w
        #print("ratio: ",ratio)
        width = int(w * ratio)
        height = int(h * ratio)

        out_list = []
        for image in input_list:
            dim = (width, height)
            image = cv2.resize(image, dim)
            out_list.append(image)

        #print("[R] out list: ",len(out_list))
        return out_list

class MultiHorizontalFlip(object):
    def __init__(self, p=0.5, angle_index=1):
        self.p = p
        self.angle_index = angle_index

    def __call__(self, input_list):
        
        #print("[HF] in list: ",len(input_list))
        out_list = []
        if random.random() < self.p:

            for image in input_list:
                image = cv2.flip(image, 1)
                out_list.append(image)

        else:
            out_list = input_list
            
        return out_list

class MultiRandomVerticalFlip(object):
    def __init__(self, p=0.5, angle_index=0):
        self.p = p
        self.angle_index = angle_index

    def __call__(self, input_list):
        
        out_list = []
        #print("[VF] in list: ",len(input_list))

        if random.random() < self.p:

            for image in input_list:
                image = cv2.flip(image, 0)
                out_list.append(image)
        else:
            out_list = input_list
        
        #print("[VF] out list: ",len(out_list)) 
        return out_list

class MultiRandomCrop(object):
    def __init__(self, size, mode='rand'):
        self.img_size = size
        if isinstance(size, int):
                self.crop_h = size
                self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))

        assert mode in ['rand', 'center']
        self.crop_type = mode

    def __call__(self, input_list):
        
        out_list = []
        #print("[RC] in list: ",len(input_list))
        h,w,c = input_list[0].shape


        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)

        for image in input_list:
            crop = image[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            out_list.append(crop)

        #print("[RC] out list: ",len(out_list))

        return out_list
