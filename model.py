"""
James Marcel
CS5330 - FInal Project
srCNN model
"""

import torch
from torch import nn
import torch.nn.functional as F

# NeuralNet is the class that outlines the neural network and processes input
# passes input through the forward method and returns the probability of each class in a tensor
class SrCNN(nn.Module):

    #outlining the different layers in our neural network
    def __init__(self):
        super(SrCNN, self).__init__()
        pad1 = 4
        pad2 = 0
        pad3 = 2

        #use padding = 'same' for inputs of any size i think
        #params are from the dong paper
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 9, padding = 'same')
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 1, padding = 'same')
        self.conv3 = nn.Conv2d(32, 3, kernel_size = 5, padding = 'same')


    def forward(self, x):

        x = torch.transpose(x,1,3)
        #print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.transpose(x,3,1)
        return x
