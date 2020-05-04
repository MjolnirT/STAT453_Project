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


class ColorizationResNet(nn.Module):
    
    def __init__(self):
        super(ColorizationResNet, self).__init__()
        
        extractor = models.resnet34(pretrained=True)
        extractor.load_state_dict(torch.load('pretrained/resnet34.pth'))
        #extractor.conv1.weight = nn.Parameter(extractor.conv1.weight.sum(dim=1).unsqueeze(1).data)

        self.extractor_net = nn.Sequential(*list(extractor.children())[0:7])
        
        #self.reverse_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        #self.bn1 = nn.BatchNorm2d(256)
        self.reverse_conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.reverse_conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.reverse_conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.reverse_conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)
        
    def forward(self, out):
        out = self.extractor_net(out)
        #out = F.relu(self.bn1(self.reverse_conv1(out)))
        #out = self.upsample(out)
        out = F.relu(self.bn2(self.reverse_conv2(out)))
        out = self.upsample(out)
        out = F.relu(self.bn3(self.reverse_conv3(out)))
        out = self.upsample(out)
        out = F.relu(self.bn4(self.reverse_conv4(out)))
        out = self.upsample(out)
        out = F.sigmoid(self.reverse_conv5(out))
        out = self.upsample(out)
        
        return out
    
class ColorizationResNet2(nn.Module):
    
    def __init__(self):
        super(ColorizationResNet2, self).__init__()
        
        extractor = models.resnet34(pretrained=True)
        extractor.load_state_dict(torch.load('pretrained/resnet34.pth'))
        #extractor.conv1.weight = nn.Parameter(extractor.conv1.weight.sum(dim=1).unsqueeze(1).data)

        self.extractor_net = nn.Sequential(*list(extractor.children())[0:6])
        
        #self.conv1 = nn.ConvTranspose2d(512, 625, kernel_size=3, stride=1, padding=1)
        #self.bn1 = nn.BatchNorm2d(625)
        #self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        #self.bn2 = nn.BatchNorm2d(256)
        #self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        #self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 500, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(500)
        self.conv_fc1 = nn.Conv2d(500, 1000, kernel_size=1, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(1000)
        self.conv_fc2 = nn.Conv2d(1000, 625, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, out):
        out = self.extractor_net(out)
        #out = F.relu(self.bn1(self.conv1(out)))
        #out = self.upsample(out)
        #out = F.relu(self.bn2(self.conv2(out)))
        #out = self.upsample(out)
        #out = F.relu(self.bn3(self.conv3(out)))
        out = self.upsample(out)
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.upsample(out)
        out = F.relu(self.bn5(self.conv5(out)))
        out = self.upsample(out)
        out = F.relu(self.bn6(self.conv_fc1(out)))
        out = F.relu(self.conv_fc2(out))
        out_prob = torch.softmax(out, dim=1)
        out_logsoftmax = self.logsoftmax(out)
        return  out_logsoftmax, out_prob
    
    
