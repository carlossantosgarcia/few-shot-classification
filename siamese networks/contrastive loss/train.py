import random
import time
import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from livelossplot import PlotLosses

from dataset import path_to_label, SiameseDataset
from loss import ContrastiveLoss, contrastive_batch_loss, mean_average_precision
from model import SiameseResnet

def enforce_all_seeds(seed):
    '''Forces seeds for reproducibility and consistancy in our results'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    rgen = np.random.default_rng(seed)
    return rgen

# Seed forced
rgen = enforce_all_seeds(42)

# Preprocessing steps applied during training/validation steps
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 250

# Dataframe used during training
ROOT = ''
train_df = pd.read_csv(ROOT+'train.csv', index_col=False, usecols=['path', 'label'])

# Validation datasets
support_givenchy = pd.read_csv('support_givenchy.csv', index_col=False, usecols=['path', 'label'])
query_givenchy = pd.read_csv('query_givenchy.csv', index_col=False, usecols=['path', 'label'])

support_versace = pd.read_csv(ROOT+'support_versace.csv', index_col=False, usecols=['path', 'label'])
query_versace = pd.read_csv(ROOT+'query_versace.csv', index_col=False, usecols=['path', 'label'])



data = SiameseDataset(train_df, # Dataframe used during training
                      length = 10000, # 10k pairs are shown to our net per epoch
                      chosen_labels = None,     
                      transform = data_transforms['train'], 
                      positive_pairs=0.5)

siamese_dataloader = DataLoader(data,
                                shuffle=False, # Images are already taken at random by construction
                                num_workers=0,
                                batch_size=batch_size)

net = SiameseResnet().to(device)

# Helpful variables for saving logs
loss_history = []
hard_examples_history = []
mean_loss_history = []
nb_batchs = data.length//batch_size
today = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
exp_name = 'resnet_10k_'
weights_file =  exp_name + str(today) +'.pt'
CSV_logs_file = exp_name + str(today) +'.csv'
df_logs = pd.DataFrame({'epoch':[], 'batch':[], 'loss':[], 'hard_loss':[], 'versace_mAP':[], 'givenchy_mAP':[] })

# Hyperparameters
EPOCHS = 50
TOP_K = int(batch_size*0.4) # Proportion of elements in each batch used to train the network
margin = 4.0 # Loss margin

criterion = ContrastiveLoss(margin)
optimizer = optim.Adam(net.parameters(),lr = 0.00005) # Original = 0.00005
groups = {'contrastive loss': ['mean loss', 'hard_examples loss', 'last batch loss'], 'mAP': ['Versace mAP', 'Givenchy mAP']}
liveloss = PlotLosses(groups=groups)

start = time.time()
for epoch in range(EPOCHS):
    mean_epoch_loss = []
    for i, batch in enumerate(siamese_dataloader):
        net.train()
        img_1, img_2 , label = batch
        img_1, img_2 , label = img_1.to(device), img_2.to(device) , label.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            # Computes the loss for each batch element, sorts the batch to get negatives examples
            temp_output_1, temp_output_2 = net(img_1,img_2)
            temp_loss_contrastive = contrastive_batch_loss(temp_output_1,temp_output_2,label,margin)
            mean_epoch_loss.append(torch.mean(temp_loss_contrastive).item())      
            loss, indexes = torch.sort(temp_loss_contrastive, dim=0)

        # Negative mining: retrieves indexes of the given % of worse examples
        indexes = indexes[-TOP_K:] 
        input_1, input_2 = img_1[indexes].squeeze(), img_2[indexes].squeeze()
        mined_labels = label[indexes].squeeze(dim=2)

        # Computes forward pass on selected pairs.
        output_1, output_2 = net(input_1,input_2)
        loss_contrastive = criterion(output_1,output_2,mined_labels)
        loss_contrastive.backward()
        optimizer.step()

        print(f"\r Epoch number {epoch+1}/{EPOCHS}, batch number {i+1}/{int(data.length/batch_size)}, current hard loss={loss_contrastive.item(): .5}, batch loss={mean_epoch_loss[-1]: .5}", end='')

        # Logs update: CSV files and weights are saved at each epoch
        net.eval()
        hard_examples_history.append(loss_contrastive)
        loss_history.append(torch.mean(temp_loss_contrastive))
        if i + 1 != nb_batchs:
            df_logs = df_logs.append(pd.DataFrame({'epoch':[epoch+1], 'batch':[i+1], 'loss':[mean_epoch_loss[-1]], 'hard_loss':[loss_contrastive.item()], 'versace_mAP':[np.nan], 'givenchy_mAP':[np.nan]}))
        else:
            versace_mAP = mean_average_precision(net, support_versace, query_versace, data_transforms)
            givenchy_mAP = mean_average_precision(net, support_givenchy, query_givenchy, data_transforms)
            df_logs = df_logs.append(pd.DataFrame({'epoch':[epoch+1], 'batch':[i+1], 'loss':[mean_epoch_loss[-1]], 'hard_loss':[loss_contrastive.item()], 'versace_mAP':[versace_mAP], 'givenchy_mAP':[givenchy_mAP]}))

    mean_loss_history.append(np.mean(mean_epoch_loss))
    df_logs.to_csv(ROOT + 'training_logs/' + CSV_logs_file, index=False)
    
    # Livelossplot logs
    liveloss.update({'mean loss': np.mean(mean_epoch_loss),
                     'hard_examples loss': loss_contrastive.item(),
                     'last batch loss': mean_epoch_loss[-1],
                     'Versace mAP': versace_mAP,
                     'Givenchy mAP': givenchy_mAP})
    liveloss.send()
    torch.save(net,ROOT+'training_logs/'+weights_file)
end = time.time()
print(f'\n Training time {end-start: .5}s, that is {int((end-start)//3600)}h{int((end-start)%3600/60)}min')