"""
CS5100 Final Project
Evaluation Mode for live input to CNN model


"""

import sys
from dataset import EvaluationESP as eval
import torch
from model import SrCNN,ESPCN
from torchvision.transforms import ToPILImage


#displaying output images
def showOutput(output, target, num):
    #array = torch.clone(example_data[2])
    tArray = target[num].permute(2,0,1)
    oArray = output[num].permute(2,0,1)


    toImage = ToPILImage()
    img = toImage(tArray)
    img2 = toImage(oArray)

    print("showing original")
    img.show()
    
    print("showing prediction")
    img2.show()
    img2.save("ESPCN_Test.png")
    print("saved output.")


def main(argv):
    #embedding for word data
    print("Loading model...")

    if (torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)

    #loading model
    torch.backends.cudnn.enabled = False
    network = ESPCN(3)

    network.load_state_dict(torch.load("ESPCN_Trained.pth"))
    network.eval()
  
    newData = eval("screen", 2)

    #loading dataset for use with loaded model
    eval_loader = torch.utils.data.DataLoader(newData, batch_size = 1, shuffle = False)
    
    #dataloader needs to be broken up into individual examples (there's only one)
    examples = enumerate(eval_loader)
    batch, (example_data, example_targets) = next(examples)

    #xexample_data now holds the feature vector/label
    output = network(example_data)

    print(output[0].shape)
    print(example_targets[0].shape)

    showOutput(output, example_targets, 0)





if __name__ == "__main__":
  main(sys.argv)