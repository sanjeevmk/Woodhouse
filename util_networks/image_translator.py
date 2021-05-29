import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

class ImageTranslator(Module):
    def __init__(self,image_size=(512,512),input_dim=3,output_dim=32):
        super(ImageTranslator,self).__init__()
        self.image_height = image_size[0]
        self.image_width = image_size[1]

        self.conv1 = nn.Conv2d(3,16,3)
        self.max1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16,32,3)
        self.max2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32,64,3)
        self.max3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64,64,1,padding=3)
        self.max4 = nn.MaxPool2d(2)

        self.linear1tx = nn.Linear(3,16)
        self.linear2tx = nn.Linear(16,32)
        self.linear3tx = nn.Linear(32,32)
        self.linear1 = nn.Linear(input_dim,256)
        self.linear2 = nn.Linear(256,512)
        self.linear3 = nn.Linear(512,512)
        self.linear4 = nn.Linear(512,output_dim)
        self.output_dim = output_dim

    def forward(self,image,texture):
        x = image.view(image.size()[0]*image.size()[1]*image.size()[2],image.size()[3])
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        #xt = torch.unsqueeze(xt,0)
        #xt = torch.repeat_interleave(xt,x.size()[0],0)
        x = torch.sigmoid(self.linear4(x))
        x = x.view(image.size()[0],image.size()[1],image.size()[2],self.output_dim)

        return x
