import torch
from pytorch3d.ops import GraphConv
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

class MeshRenderer(Module):
    def __init__(self,image_size=(512,512),input_dim=3):
        super(MeshRenderer,self).__init__()
        self.image_height = image_size[0]
        self.image_width = image_size[1]

        self.graph_layer1 = GraphConv(input_dim=input_dim,output_dim=256)
        self.graph_layer2 = GraphConv(input_dim=256,output_dim=512)
        self.graph_layer3 = GraphConv(input_dim=128,output_dim=128)
        self.graph_layer4 = GraphConv(input_dim=128,output_dim=3*128*128)

        self.linear0 = nn.Linear(input_dim,16)
        self.linear1 = nn.Linear(16,32)

        self.linear2 = nn.Linear(512,1024)
        self.linear3 = nn.Linear(1024,3*128*128)
        self.upsample = nn.Upsample(scale_factor=4)

    def forward(self,verts,edges):
        x = F.relu(self.graph_layer1(verts,edges))
        x = F.relu(self.graph_layer2(x,edges))
        x,_ = torch.max(x,0)
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        image = x.view(-1,3,self.image_height,self.image_width)
        image = self.upsample(image)
        return image