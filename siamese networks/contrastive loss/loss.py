import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score

class ContrastiveLoss(nn.Module):
    ''' Implements contrastive loss to train Siamese Network'''

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_1, output_2, label):
        '''Computes the average contrastive loss for a given batch'''

        # Distance between embedded outputs
        L2_distance = F.pairwise_distance(output_1, output_2, keepdim = True)
        
        # Loss calculation
        losses = (1-label) * torch.pow(L2_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - L2_distance, min=0.0), 2)
        contrastive_loss = torch.mean(losses)

        return contrastive_loss

def contrastive_batch_loss(output_1, output_2, label, margin=2.0 ):
    ''' Computes the loss for each pair of images in the batch, not returning mean but a tensor with losses to sort them afterwards'''
    L2_distance = F.pairwise_distance(output_1, output_2, keepdim = True)
    contrastive_loss = (1-label) * torch.pow(L2_distance, 2) + (label) * torch.pow(torch.clamp(margin - L2_distance, min=0.0), 2)
    return contrastive_loss

def mean_average_precision(net, support_df, query_df, custom_transforms):
    '''Computes mAP for the given network on support and query datasets'''
    transformer = custom_transforms['val']

    def preprocess(path):
        '''Returns image in tensor formed ready to be fed to Siamese network'''
        return transformer(Image.open(path).convert("RGB"))

    def forward_pass(path, net):
        '''Performs a forward pass into the net'''
        img = preprocess(path)
        y = net.forward_one(img.unsqueeze(0).cuda())
        y = y.detach().cpu().numpy()
        return y

    support_df['embedded_images'] = support_df.path.apply(lambda x:forward_pass(x, net))
    query_df['embedded_images'] = query_df.path.apply(lambda x:forward_pass(x, net))

    def calculate_AP(label):
        '''calculates AP for the given label'''
        # Ground truth vector
        y_ground = support_df.label.apply(lambda x: 1 if x==label else 0).values
        
        # Embedded query
        img_embedded = query_df.embedded_images.loc[query_df.label == label].values[0]
                
        def distance_to_query(x):
            return -np.linalg.norm(x-img_embedded)
        
        # Distances vector
        y_distances = support_df.embedded_images.apply(lambda x: distance_to_query(x))
        return average_precision_score(y_ground, y_distances)

    query_df['AP'] = query_df.label.apply(lambda x:calculate_AP(x))
    mAP = np.mean(query_df['AP'].values)
    return mAP