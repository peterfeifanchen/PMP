import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image

class BlinkDataSet(data.Dataset):
    def __init__(self, xFilePaths, yLabels):
        self.xFilePaths = xFilePaths
        self.yLabels = yLabels

        self.xImages = [ Image.open(path) for path in self.xFilePaths ]
        self.yTensors = [ torch.Tensor( [float(yValue)] ) for yValue in self.yLabels ]

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.], std=[0.5])
            ])
    
    def __len__(self):
        return len(self.xImages)

    def __getitem__(self, index):
        x = self.transform(self.xImages[index])
        y = self.yTensors[index]

        return x,y

class LeNet(torch.nn.Module):
    def __init__(self, imageSize, convFilters, fcLayers):
        super(LeNet, self).__init__()
        in_channels = 1
        imageOutSize = imageSize
        self.convLayers = torch.nn.ModuleList()
        self.hiddenLayers = torch.nn.ModuleList()
        for convFilter in convFilters:
            self.convLayers.extend(  torch.nn.Sequential(
                torch.nn.Conv2d( in_channels=in_channels, out_channels=convFilter[0], kernel_size=convFilter[1]),
                torch.nn.ReLU(), 
                torch.nn.AvgPool2d(kernel_size = 2, stride = 2)
            ))
            in_channels=convFilter[0]
            imageOutSize = (imageOutSize - convFilter[1]) + 1
            imageOutSize = int((imageOutSize-2)/2 + 1)

        in_size = in_channels * imageOutSize * imageOutSize
        for fcLayer in fcLayers:
            self.hiddenLayers.extend( torch.nn.Sequential(
                torch.nn.Linear( in_size, fcLayer ),
                torch.nn.ReLU()
            ))
            in_size = fcLayer

        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(fcLayers[1], 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        out = x
        for convlayer in self.convLayers:
            out = convlayer(out)
        
        out = out.reshape(out.size(0), -1)

        for hiddenlayer in self.hiddenLayers:
           out = hiddenlayer(out)
      
        return self.outputLayer(out)

class BlinkNeuralNetwork(torch.nn.Module):
    def __init__(self, hiddenNodes = [5, 5]):
        super(BlinkNeuralNetwork, self).__init__()

        # Down sample the image to 12x12
        self.avgPooling = torch.nn.AvgPool2d(kernel_size = 2, stride = 2) 

        # Fully connected layer from all the down-sampled input pixels to all the hidden nodes
        self.fullyConnectedOne = torch.nn.Sequential(
           torch.nn.Linear(12*12, hiddenNodes[0]),
           torch.nn.Sigmoid()
           )

        self.fullyConnectedTwo = torch.nn.Sequential(
            torch.nn.Linear(hiddenNodes[0], hiddenNodes[1]),
            torch.nn.Sigmoid()
        )

        # Fully connected layer from the hidden layer to a single output node
        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(hiddenNodes[1], 1),
            torch.nn.Sigmoid()
            )

    def forward(self, x):
        # Apply the layers created at initialization time in order
        
        out = self.avgPooling(x)
        
        # This turns the 2d samples into 1d arrays for processing with later layers
        out = out.reshape(out.size(0), -1)
        
        out = self.fullyConnectedOne(out)
        out = self.fullyConnectedTwo(out)
        out = self.outputLayer(out)

        return out