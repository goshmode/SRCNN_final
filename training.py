""" James Marcel
    CS5330 Final Project

    Single Image Super-Resolution using CNN. 
    Training/validation structure for SRCNN and ESPCN.
"""

import torch 
from torchvision.transforms import ToPILImage
from torchmetrics import PeakSignalNoiseRatio
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from dataset import srSet, espcnSet
import ignite.metrics as im
from statistics import mean
from model import SrCNN, ESPCN
import csv



#training function
#sets the network to train mode and runs the optimizer based on the MSE loss from the forward pass
def train(epoch,network,optimizer,train_loader,log_interval,train_losses, train_counter, device):
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
        #calculate loss (MSE)
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
            #saving states of network
            torch.save(network.state_dict(), 'SRCNN.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')


data_master = []
globalCount = 0
#test function
# runs through test data with the network in eval mode (doesn't change weights)
def test(network, test_loader, test_losses, device):

    network.eval()
    test_loss = 0

    #for validation set stats
    psnrList = []
    ssimList = []
    SSIM = im.SSIM(data_range = 1, kernel_size = 11)
    SSIMBi = im.SSIM(data_range = 1, kernel_size = 11)
    PSNR = PeakSignalNoiseRatio()
    SSIMBiList = []
    PSNRList2 = []

    #test loop
    with torch.no_grad():
        for data, target in test_loader:
            #taking advantage of cuda
            data = data.to(device)
            #run convolution
            output = network(data)
            #move output to cpu
            output = output.to('cpu')

            #Dong paper uses MSE for loss function
            lossFn = nn.MSELoss()
            test_loss += lossFn(output, target)
            
            #getting right shape for SSIM
            scaledData = torch.transpose(data,1,3)

            #for calculating PSNR and SSIM on bicubic interpolation
            scaledData2 = nn.functional.interpolate(scaledData, scale_factor = 3, mode = 'bicubic')
            scaledData2 = scaledData2.to('cpu')
            
            output2 = torch.transpose(output,1,3)
            target2 = torch.transpose(target,1,3)
            print(scaledData2.shape, target2.shape)
            psnr = PSNR(output, target)
            psnr2 = PSNR(scaledData2,target2)

            SSIMBi.update((scaledData2,target2))
            SSIM.update((output2, target2))

            #storing some stats. 
            #print(f"PSNR is {psnr} and SSIM is {SSIM.compute()}")
            psnrList.append(psnr.item())
            ssimList.append(SSIM.compute())
            PSNRList2.append(psnr2.item())
            SSIMBiList.append(SSIMBi.compute())
    
    data_master.append(mean(psnrList))
    data_master.append(mean(ssimList))
    print(f"Bicubic results were:  psnr: {mean(PSNRList2)}  SSIM: {mean(SSIMBiList)}")
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f"\nTest set: Avg. loss: {test_loss:.4f}, PSNR avg: {mean(psnrList)} SSIM avg: {mean(ssimList)} \n")
    #returning loss to decide when to stop training
    return test_loss


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
    n_epochs = 200  #5 epochs shouldn't take too too long. each epoch runs through all of the training/test data
    batch_size_train = 32
    batch_size_test = 32
    learning_rate = 0.0003
    momentum = 0.5
    log_interval = 16

    if (torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)

    torch.manual_seed(42)

    #loading data to datasets
    training_images = srSet("./data/TrainSet.csv","train")
    test_images = srSet("./data/TestSet.csv","test")

    training_images = espcnSet("./data/TrainSet.csv","train")
    test_images = espcnSet("./data/TestSet.csv","test")

    #Loading Training data
    # The data needs to be passed to our neural network as a DataLoader class instance
    train_loader = torch.utils.data.DataLoader(training_images, batch_size = batch_size_train, shuffle = False)

    #Loading Test data
    test_loader = torch.utils.data.DataLoader(test_images,batch_size = batch_size_test, shuffle = False)

    print(f"loaded up the Images. Train loader: {len(train_loader)} Test Loader: {len(test_loader)}\n")


    #initializing network and optimizer - un/comment to change which model to train
    #network = SrCNN()
    network = ESPCN(3)
    network.to(device)
    #used Adam optimizer
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    #saving accuracy data to these lists for plotting later
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    ##### TRAINING/TEST LOOP ######
    
    #Training and testing the datasets for n_epochs times
    test(network, test_loader,test_losses,device)
    for epoch in range(1,n_epochs + 1):
        train(epoch, network,optimizer, train_loader, log_interval, train_losses, train_counter, device)
        test(network, test_loader, test_losses, device)

    
    with open('ESPCN_stats', 'w') as f:
        write = csv.writer(f)
        for x in range(0,len(data_master) - 1, 2):
            print(x)
            write.writerow([data_master[x],data_master[x+1]])


    #saving model
    modelSave(network, "ESPCN_Trained.pth")
    
    



if __name__ == "__main__":
    main(sys.argv)