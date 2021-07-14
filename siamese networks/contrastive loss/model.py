import torch.nn as nn
import torchvision.models as models 
import torch.nn.functional as F

class SiameseResnet(nn.Module):
    ''' Siamese network to learn images representation'''
    def __init__(self):
        super(SiameseResnet, self).__init__()
        
        # Loading ResNet
        model = models.resnet18(pretrained=True)
            
        # Removing last fully-connected layer
        model.fc = nn.Sequential()
        self.extractor = model
        
        # Fully-connected layers
        self.fc1 = nn.Sequential(nn.Linear(512,512),nn.LeakyReLU())
        self.fc2 = nn.Linear(512,256)

    def forward_one(self, x):
        ''' Computes the output of the model for a single image'''
        x = self.extractor(x)
        x = self.fc1(x)
        x = F.sigmoid(self.fc2(x))
        return x

    def forward(self, x1, x2):
        ''' Returns a pair of outputs'''
        output_1 = self.forward_one(x1)
        output_2 = self.forward_one(x2)
        return output_1, output_2