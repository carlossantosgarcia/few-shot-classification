import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

def path_to_label(path, normal = False):
    '''Computes the label from the path of each image. Needs to be consistent with the labels in the csv files.'''
    if normal:
        path = path.split('/')
    else:
        path = path.split('\\')
    label = path[2] # Remove the first two folders in which our images lie
    for w in path[3:-1]:
        if not w.endswith('.jpg'):
            label = label + '-' + w        
    return label

class SiameseDataset(Dataset):
    ''' Implements dataset creation for siamese network.'''

    def __init__(self,df,length,chosen_labels=None,transform=None, positive_pairs=0.5):
        self.df = df # Dataframe needs at least the columns "path" and "label"
        self.len_df = len(df)    
        if chosen_labels is not None:
            self.chosen_labels = chosen_labels
        else:
            self.chosen_labels = df_paths_labels.label.unique()
        self.length = length
        self.transform = transform
        self.positive_pairs = positive_pairs # Proportion of positive pairs fed during training
        
    def __getitem__(self,index):
        '''Selects first label at random. Second image depends on positive pairs proportion wanted'''
        path_1 = random.choice(self.df.loc[self.df.label.isin(self.chosen_labels)].path.values)
        label_1 = path_to_label(path_1)
        # Dataset with a fraction p of positively labeled pairs
        same_label = random.random()
        same_label = int(same_label < self.positive_pairs)

        if same_label:
            # Picks image from the same label as the first one
            path_2 = random.choice(self.df.loc[(self.df.label == label_1) & ~(self.df.path == path_1)].path.values)
            y = torch.from_numpy(np.array([0],dtype=np.float32))
        else:
            # Picks image from a different label
            path_2 = random.choice(self.df.loc[(self.df.label != label_1)].path.values)
            y = torch.from_numpy(np.array([1],dtype=np.float32))

        img_1 = Image.open(path_1).convert("RGB")
        img_2 = Image.open(path_2).convert("RGB")

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        
        return img_1, img_2 , y
    
    def __len__(self):
        return self.length