import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(Module):
    def __init__(self,image_size=(1024,1024),input_dim=3,output_dim=32):
        super(ImageEncoder,self).__init__()
        self.image_height = image_size[0]
        self.image_width = image_size[1]

        self.conv1 = nn.Conv2d(3,16,4)
        self.max1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16,32,4)
        self.max2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32,64,4)
        self.max3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64,128,4)
        self.max4 = nn.MaxPool2d(2)

        self.linear1 = nn.Linear(128,output_dim)

    def forward(self,image):
        x = F.relu(self.conv1(image) )
        x = self.max1(x)
        x = F.relu(self.conv2(x) )
        x = self.max2(x)
        x = F.relu(self.conv3(x) )
        x = self.max3(x)
        x = F.relu(self.conv4(x) )
        x = self.max4(x)
        x = F.max_pool2d(x,kernel_size=x.size()[2:])
        x = torch.unsqueeze(torch.squeeze(x),0)
        x = torch.sigmoid(self.linear1(x))
        return x
