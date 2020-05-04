import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

from skimage.color import rgb2lab, lab2rgb, rgb2gray
import matplotlib.pyplot as plt
import cv2  
import os
from PIL import Image

class ColorizationDataset(datasets.ImageFolder):
    '''Custom images folder, which converts images to grayscale before loading'''
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original).transpose(1,2,0) #change format from NCHW into NHWC
            img_lab = rgb2lab(img_original)
            img_ab = img_lab[:,:,1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))) #changing format back into NCHW
            #img_lab = (img_lab + 128) / 255 #rescaling into [0,1], only considering ab channels
            img_original = rgb2gray(img_original)
            img_original = np.expand_dims(img_original, axis=0)
            img_original = np.concatenate((img_original,img_original,img_original),axis=0)
            img_original = torch.from_numpy(img_original).float()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_original, img_ab, target

    
def compute_mse(net, data_loader, device):
    '''compute mse of whole trainning set or valid set'''
    curr_mse, num_examples = torch.zeros(1).float().to(DEVICE), 0
    with torch.no_grad():
        for input_gray, input_ab, targets in data_loader:
            input_gray = input_gray.to(DEVICE)
            input_ab = input_ab.to(DEVICE)

            output_ab = net.forward(input_gray)
            loss = torch.sum((output_ab - input_ab)**2, dim=(0,1,2,3))
            num_examples += targets.size(0)
            curr_mse += loss

        curr_mse = torch.mean(curr_mse/num_examples, dim=0)
        return curr_mse        
    
def compute_cross_entropy(net, data_loader, device):
    '''compute cross_entropy of whole trainning set or valid set'''
    curr_ce, num_examples = torch.zeros(1).float().to(DEVICE), 0
    with torch.no_grad():
        for input_gray, input_ab, targets in data_loader:
            input_gray = input_gray.to(DEVICE)
            input_ab = input_ab.to(DEVICE)
            
            output_ab,_ = net.forward(input_gray)
            input_ab = conti_ab_2_class(input_ab)
            input_ab = torch.from_numpy(input_ab).long().to(DEVICE)
            
            loss = nn.CrossEntropyLoss()(output_ab, input_ab)
            num_examples += 1
            curr_ce += loss
            
        curr_ce = torch.mean(curr_ce/num_examples, dim=0)
        return curr_ce
    
def conti_ab_2_class(input_ab_conti):
    '''quantize continues values in ab channel into class number'''
    #input  NCHW batch*2(ab channel)*height*weight
    #output NHW batch*height*weight int32 data type
    input_ab_conti += 128
    input_ab_conti = input_ab_conti/10
    input_ab_conti = input_ab_conti.to('cpu').numpy().astype(int)
    input_a = input_ab_conti[:,0,:,:]
    input_b = input_ab_conti[:,1,:,:]
    input_ab_class = input_a*25 + input_b
    
    return input_ab_class

def class_2_ab(output_ab,W):
    '''transfere 625 channels with its probility into two ab channels'''
    #input   NCHW batch*625*height*weight, the value is the prob of class assigned to the pixel
    #output  NCHW  batch*2(ab channel)*height*weight
    out_ab_max = np.argmax(output_ab, axis=1)
    output_a = (out_ab_max/25)*10
    output_b = np.mod(out_ab_max, 25)*10

    output_a = np.expand_dims(output_a, axis=1)
    output_b = np.expand_dims(output_b, axis=1)
    output_ab_new = np.concatenate((output_a, output_b), axis=1)
    
    return output_ab_new