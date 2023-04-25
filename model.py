"""
James Marcel
CS5330 - FInal Project
srCNN model
"""

import torch
from torch import nn
import torch.nn.functional as F

# Model proposed by Dong et al.
# Takes low-res image scaled up (bicubic) to hi-res size and learns based on that 
class SrCNN(nn.Module):

    #outlining the different layers in our neural network
    def __init__(self):
        super(SrCNN, self).__init__()

        #use padding = 'same' to keep output same size
        #params are from the dong paper - 64 layers for 1st conv, then 32
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 9, padding = 'same')
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 1, padding = 'same')
        self.conv3 = nn.Conv2d(32, 3, kernel_size = 5, padding = 'same')


    def forward(self, x):
        #channel order isn't as expected, so switching color channel with a spatial dim
        x = torch.transpose(x,1,3)
        #print(x.shape)
        #run relu after first two convolution layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        #clamp values between 0 and 1, and transpose to channel in 2nd dim
        x = torch.clamp_(x, 0.0, 1.0)
        x = torch.transpose(x,3,1)
        return x



# model proposed by Shi et al
# takes scaled down input and learns upscaling filter directly
# requires a scaling factor (3 is used for training, but can be more for eval)
class ESPCN(nn.Module):

    #outlining the different layers in our neural network
    def __init__(self, scale):
        super(ESPCN, self).__init__()
        self.upscaling = scale
        #use padding = 'same' to keep output same size
        #params are from the dong paper - 64 layers for 1st conv, then 32
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 9, stride = 1,padding = 'same')
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 1, stride = 1, padding = 'same')
        self.out = int(3 * (self.upscaling ** 2))

        self.conv3 = nn.Conv2d(32, self.out, kernel_size = 5, stride = 1, padding = 'same')

        #learned upsampling layer
        self.subPix = nn.PixelShuffle(self.upscaling)
        #adjusted scale for input to SubPix
        

    def forward(self, x):
        #channel order isn't as expected, so switching color channel with a spatial dim
        x = torch.transpose(x,1,3)
        #print(x.shape)
        #run relu after first two convolution layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        #pixel shuffle learned upsampling
        x = self.subPix(x)
        #clamping values between 0 and 1
        x = torch.clamp_(x, 0.0, 1.0)
        x = torch.transpose(x,3,1)
        return x