""" James Marcel
    CS5330 Final Project

    SRCNN - Super-resolution model
"""

import torch 
from torchvision.transforms import ToPILImage, Normalize
from torchmetrics import PeakSignalNoiseRatio
import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import srSet
import ignite.metrics as im
from statistics import mean
from model import SrCNN




#######  CLASS DEFINITIONS ################
"""
# NeuralNet is the class that outlines the neural network and processes input
# passes input through the forward method and returns the probability of each class in a tensor
class SrCNN(nn.Module):

    #outlining the different layers in our neural network
    def __init__(self):
        super(SrCNN, self).__init__()

        #use padding = 'same' for inputs of any size i think
        #params are from the dong paper 
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 9, padding = 4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 1, padding = 0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size = 5, padding = 2)


    def forward(self, x):

        x = torch.transpose(x,1,3)
        #print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.transpose(x,3,1)
        return x
"""

#training function
#sets the network to train mode and runs the optimizer based on the loss from the forward pass
def train(epoch,network,optimizer,train_loader,log_interval,train_losses, train_counter, device):
    #network = network.to(device)
    network.train()
    #for each batch of our training data
    for batch_idx, (data,target) in enumerate(train_loader):
        network.to(device)
        optimizer.zero_grad()
        #run data throuh the neural network
        data = data.to(device)
        output = network(data)
        #network = network.to('cpu')
        output = output.to(device)
        target = target.to(device)
        #calculate loss
        lossFn = nn.MSELoss()
        loss = lossFn(output, target)
        #change weights accordingly
        loss.backward()
        optimizer.step()

        #this stuff is just for logging the learning process and saving the model to disk
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), 'SRCNN.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')


#test function
# runs through test data with the network in eval mode (doesn't change weights)
def test(network, test_loader,test_losses,device):

    network.eval()
    test_loss = 0
    psnrList = []
    ssimList = []
    SSIM = im.SSIM(data_range = 1, kernel_size = 11)
    PSNR = PeakSignalNoiseRatio()

    with torch.no_grad():
        for data, target in test_loader:
            #taking advanctage of cuda
            data = data.to(device)
            #run convolution
            output = network(data)
            #move output to cpu
            output = output.to('cpu')
            #Dong paper uses MSE for loss function
            lossFn = nn.MSELoss()
            test_loss += lossFn(output, target)
            psnr = PSNR(output, target)
            #getting right shape for SSIM
            output2 = torch.transpose(output,1,3)
            target2 = torch.transpose(target,1,3)
            SSIM.update((output2, target2))
            #print(f"PSNR is {psnr} and SSIM is {SSIM.compute()}")
            psnrList.append(psnr.item())
            ssimList.append(SSIM.compute())

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f"\nTest set: Avg. loss: {test_loss:.4f}, PSNR avg: {mean(psnrList)} SSIM avg: {mean(ssimList)} \n")


#plot test images
def testPlot(example_data, example_targets):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap = 'gray', interpolation = 'none')
        plt.title(f"Ground Truth: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()


#plotting model performance
def plot(train_counter, train_losses, test_counter, test_losses):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color = "green")
    print(len(test_counter), len(test_losses))
    plt.scatter(test_counter, test_losses, color = "blue")
    plt.legend(["Train Loss", "Test Loss"], loc = "upper right")
    plt.xlabel("Number of Training Examples Seen")
    plt.ylabel("Negative Log Likelihood Loss")
    plt.show()



#displaying output images
def showOutput(output, target, num):

    #array = torch.clone(example_data[2])
    tArray = target[num].permute(2,0,1)
    oArray = output[num].permute(2,0,1) 

    toImage = ToPILImage()
    img = toImage(tArray)
    img2 = toImage(oArray)
    img.show()
    img2.show()

#Save model to a file
def modelSave(model, name):
    torch.save(model.state_dict(), name)
    print("Network Saved as ", name)



#Main function 
def main(argv):

    #setting a few variables for how we train 
    n_epochs = 30  #5 epochs shouldn't take too too long. each epoch runs through all of the training/test data
    batch_size_train = 20
    batch_size_test = 20
    learning_rate = 0.0001
    momentum = 0.5
    log_interval = 10

    if (torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)

    torch.manual_seed(42)

    training_images = srSet("./data/TrainSet.csv","train")
    test_images = srSet("./data/TestSet.csv","test")


    #Loading Training data
    # The data needs to be passed to our neural network as a DataLoader class instance
    train_loader = torch.utils.data.DataLoader(training_images, batch_size = batch_size_train, shuffle = False)

    #Loading Test data
    test_loader = torch.utils.data.DataLoader(test_images,batch_size = batch_size_test, shuffle = False)

    print(f"loaded up the Images. Train loader: {len(train_loader)} Test Loader: {len(test_loader)}\n")


    #examples = enumerate(test_loader)
    #batch, (example_data, example_targets) = next(examples)


    #initializing network and optimizer
    network = SrCNN()
    network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    #saving accuracy data to these lists for plotting later
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    ##### TRAINING/TEST LOOP ######
    
    #Training and testing the datasets for n_epochs times
    #test(network, test_loader,test_losses)
    for epoch in range(1,n_epochs + 1):
        train(epoch, network,optimizer, train_loader, log_interval, train_losses, train_counter, device)
        test(network, test_loader, test_losses, device)

    """
    #plotting stuff
    plot(train_counter, train_losses, test_counter, test_losses)
    """

    #saving model
    modelSave(network, "SRCNN_Trained.pth")
    
    



if __name__ == "__main__":
    main(sys.argv)