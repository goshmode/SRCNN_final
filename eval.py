"""
CS5100 Final Project
Evaluation Mode for live input to CNN model


"""

import sys
from dataset import Evaluation as eval
import torch
from model import SrCNN
from torchvision.transforms import ToPILImage


#displaying output images
def showOutput(output, target, num):
    print("working on arrays")
    #array = torch.clone(example_data[2])
    #switching RGB channel from 3rd position to first
    tArray = target[num].permute(2,0,1)
    oArray = output[num].permute(2,0,1)


    toImage = ToPILImage()
    img = toImage(tArray)
    img2 = toImage(oArray)

    print("showing original")
    img.show()
    
    print("showing prediction")
    img2.show()
    img2.save("SRCNN_Test.png")


#loads SRCNN model, passes input file to dataset/model and then displays Hi res then Lo res.
#saves low res version
def main(argv):
    #embedding for word data
    print("Loading model...")

    if (torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)

    #loading model
    network = SrCNN().to(device)

    network.load_state_dict(torch.load("SRCNN_200.pth"))
    network.eval()


    #### FILENAME assumes there are two files, *LOW.png and *HI.png.  ex. screenLOW.png screenHI.png
    #### should also be in a folder called 'data/evaluation/'
    newData = eval("screen", 3)
    print("testing")
    #loading dataset for use with loaded model
    eval_loader = torch.utils.data.DataLoader(newData, batch_size = 1, shuffle = False)
    
    #dataloader needs to be broken up into individual examples (there's only one)
    examples = enumerate(eval_loader)
    batch, (example_data, example_targets) = next(examples)
    print("going into network.")

    #xexample_data now holds the feature vector/label
    example_data = example_data.to(device)
    output = network(example_data)
    output = output.to('cpu')
    print("got past network.")
    #print(output[0].shape)
    #print(example_targets[0].shape)

    showOutput(output, example_targets, 0)





if __name__ == "__main__":
  main(sys.argv)