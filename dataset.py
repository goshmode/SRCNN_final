'''
James Marcel
CS 5330 Final Project
SRCNN Datasets & image preprocessing

'''

import torch
import os
import pandas as pd
import numpy as np
from random import randrange
from PIL import Image, ImageFilter
from torch.utils.data import Dataset




# Dataset for images in hiRes and loRes folders. assumes matching indices for matching hi/lo images
class srSet(Dataset):
  def __init__(self, csv_file, type): 

    self.csvIn = pd.read_csv(csv_file, sep="\n", header = None)
    self.list = [x for x in self.csvIn[0].values]
    self.sampleSize = 99
    self.smallSize = 99//3
    self.type = type

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):


    hiFile = "./data/" + self.type + "/hiRes/" + str(self.list[idx])
    loFile = "./data/" + self.type + "/loRes/" + str(self.list[idx])
    #open image for this index in file list
    hi = Image.open(hiFile)
    lo = Image.open(loFile)
    lo = lo.resize((self.sampleSize, self.sampleSize))
    
	#normalizing into np arrays
    hiMat = np.asarray(hi) / 255
    loMat = np.asarray(lo) / 255
    #values = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)

	#converting to tensors
    label = torch.Tensor(hiMat)
    data = torch.Tensor(loMat)

    
    return data, label

# Preprocessing for image - saves hi and lo res images in their specific folders with matching filenames
class preProc(Dataset):
  def __init__(self, csv_file, type): 

    self.csvIn = pd.read_csv(csv_file, sep="\n", header = None)
    self.list = [x for x in self.csvIn[0].values]
    self.sampleSize = 99
    self.smallSize = 99//3
    self.type = type

  def __len__(self):
    return len(self.list)

  def __getitem__(self, idx):

    filename = "./data/DIV2K_train_HR/" + str(self.list[idx])
    #open image for this index in file list
    image = Image.open(filename)
    
    #values = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)

    #for storing rotated images
    #path = 'C:/Users/HeyDude/Documents/CS5330/data/hiRes/' + idx + ".png"
    x,y = image.size

    for i in range(8):
        #file paths
        hiPath = "C:/Users/HeyDude/Documents/CS5330/data/"+ self.type + "/hiRes/" + str(idx) + "sub" + str(i) + ".png"
        loPath = "C:/Users/HeyDude/Documents/CS5330/data/" + self.type + "/loRes/" + str(idx) + "sub" + str(i) + ".png"

        #cropping a random 33x33 sample from image
        cropX = randrange(0, x - self.sampleSize)
        cropY = randrange(0, y - self.sampleSize)
        box = (cropX, cropY, cropX + self.sampleSize, cropY + self.sampleSize)
        sample = image.crop(box)
        #rotating 1/4
        if i == 2 or i == 3:
          sample.transpose(Image.ROTATE_90)
        #flipping 1/4
        if i == 4 or i == 5:
          sample.transpose(Image.FLIP_TOP_BOTTOM)

        sample.save(hiPath, 'PNG')

        #Dong paper mentions running inputs through a gaussian blur
        small = sample.filter(ImageFilter.GaussianBlur(radius = 2))
        #resize lo res to 3x smaller
        small = small.resize((self.smallSize, self.smallSize))

        small.save(loPath, 'PNG')

    return 0, 1
  

class Evaluation(Dataset):
  def __init__(self, filename, scale):

    self.filename = filename
    self.scale = scale

  def __len__(self):
    return 1

  def __getitem__(self, idx):


    hiFile = "C:/Users/HeyDude/Documents/CS5330/data/evaluation/" + str(self.filename) + "HI.png"
    loFile = "C:/Users/HeyDude/Documents/CS5330/data/evaluation/" + str(self.filename) + "LOW.png"

    #open image for this index in file list
    lo = Image.open(loFile)
    height = lo.size[0] * self.scale
    width = lo.size[1] * self.scale
    lo = lo.resize((height,width))

    hi = Image.open(hiFile)
    
	#normalizing into np arrays
    loMat = np.asarray(lo) / 255
    hiMat = np.asarray(hi) / 255

    print(loMat.shape)

	#converting to tensors
    label = torch.Tensor(hiMat)
    data = torch.Tensor(loMat)

    
    return data, label





if __name__ == "__main__": 

  #loading a test dataset to check for errors
  #tweet_data = OlidTrainingDataset("./data/olid_training_set_google_300.csv", vector_size=300)
  string = "test"
  test = preProc("./data/testset_filenames.csv", string)
  #test = preProc("./data/test_filenames.csv")
  #print (test[0][1].shape)

  for i in range(len(test)):
    img = test[i]


  print("Loaded filenames.")