#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This script contains the helper functions you will be using for this assignment
import os
import random

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[4]:


class BassetDataset(Dataset):
    """
    BassetDataset class taken with permission from Dr. Ahmad Pesaranghader

    We have already processed the data in HDF5 format: er.h5
    See https://www.h5py.org/ for details of the Python package used

    We used the same data processing pipeline as the paper.
    You can find the code here: https://github.com/davek44/Basset
    """

    # Initializes the BassetDataset
    def __init__(self, path='./data/', f5name='er.h5', split='train', transform=None):
        """
        Args:
            :param path: path to HDF5 file
            :param f5name: HDF5 file name
            :param split: split that we are interested to work with
            :param transform (callable, optional): Optional transform to be applied on a sample
        """

        self.split = split

        split_dict = {'train': ['train_in', 'train_out'],
                      'test': ['test_in', 'test_out'],
                      'valid': ['valid_in', 'valid_out']}

        assert self.split in split_dict, "'split' argument can be only defined as 'train', 'valid' or 'test'"

        # Open hdf5 file where one-hoted data are stored
        self.dataset = h5py.File(os.path.join(path, f5name.format(self.split)), 'r')

        # Keeping track of the names of the target labels
        self.target_labels = self.dataset['target_labels']

        # Get the list of volumes
        self.inputs = self.dataset[split_dict[split][0]]
        self.outputs = self.dataset[split_dict[split][1]]

        self.ids = list(range(len(self.inputs)))
        if self.split == 'test':
            self.id_vars = np.char.decode(self.dataset['test_headers'])

    def __getitem__(self, i):
        """
        Returns the sequence and the target at index i

        Notes:
        * The data is stored as float16, however, your model will expect float32.
          Do the type conversion here!
        * Pay attention to the output shape of the data.
          Change it to match what the model is expecting
          hint: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        * The target must also be converted to float32
        * When in doubt, look at the output of __getitem__ !
        """

        idx = self.ids[i]

        # Sequence & Target
        output = {'sequence': None, 'target': None}

        # WRITE CODE HERE
        target = self.outputs[idx]
        target = torch.from_numpy(np.array(target))
        target = target.type(torch.float32)
        
        input_data = self.inputs[idx, :, :, :]
        input_data = torch.from_numpy(np.array(input_data))
        input_data = input_data.type(torch.float32)
        input_data = input_data.permute(1,2,0)
        
        output = {'sequence': input_data, 'target': target}
        return output

    def __len__(self):
        # WRITE CODE HERE
        return len(self.outputs)

    def get_seq_len(self):
        """
        Answer to Q1 part 2
        """
        # WRITE CODE HERE
        seq_len = len(self.inputs[0][0][0])
        return seq_len

    def is_equivalent(self):
        """
        Answer to Q1 part 3
        """
        # WRITE CODE HERE
        transform_sequence = self.inputs[0].reshape(-1,600,4)
        seq_len = len(self.inputs[0][0][0])
        return transform_sequence.shape == (1,seq_len,4)


# In[3]:


class Basset(nn.Module):
    """
    Basset model
    Architecture specifications can be found in the supplementary material
    You will also need to use some Convolution Arithmetic
    """

    def __init__(self):
        super(Basset, self).__init__()

        self.dropout = 0.3  # should be float
        self.num_cell_types = 164

        self.conv1 = nn.Conv2d(1, 300, (19, 4), stride=(1, 1), padding=(9, 0))
        self.conv2 = nn.Conv2d(300, 200, (11, 1), stride=(1, 1), padding=(5, 0))
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(4, 0))

        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))

        self.fc1 = nn.Linear(13*200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, self.num_cell_types)

    def forward(self, x):
        """
        Compute forward pass for the model.
        nn.Module will automatically create the `.backward` method!

        Note:
            * You will have to use torch's functional interface to 
              complete the forward method as it appears in the supplementary material
            * There are additional batch norm layers defined in `__init__`
              which you will want to use on your fully connected layers
        """

        # WRITE CODE HERE
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.maxpool3(out)
        out = torch.flatten(out, 1)
        out = F.dropout(F.relu(self.bn4(self.fc1(out))), self.dropout, self.training)
        out = F.dropout(F.relu(self.bn5(self.fc2(out))), self.dropout, self.training)
        out = self.fc3(out)
        return out


# In[ ]:


def compute_fpr_tpr(y_true, y_pred):
    """
    Computes the False Positive Rate and True Positive Rate
    Args:
        :param y_true: groundtruth labels (np.array of ints [0 or 1])
        :param y_pred: model decisions (np.array of ints [0 or 1])

    :Return: dict with tpr, fpr (values are floats)
    """
    output = {'fpr': 0., 'tpr': 0.}

    # WRITE CODE HERE

    TP = ((y_true * y_pred) == 1).sum()
    output['tpr'] = TP/ ((y_true) == 1).sum()
    
    
    converted_y_true= np.copy(y_true)
    converted_y_true[converted_y_true==1] = 5
    converted_y_true[converted_y_true==0] = 1
    converted_y_true[converted_y_true==5] = 0
    FP= ((converted_y_true * y_pred)== 1).sum()
    output['fpr'] = FP/ ((y_true) == 0).sum()
    

    return output


def compute_fpr_tpr_dumb_model():
    """
    Simulates a dumb model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds (k)
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05
                 ...
                 output['fpr_list'][-1] corresponds to k=0.95

            Do the same for output['tpr_list']
             
    """
    output = {'fpr_list': [], 'tpr_list': []}

    # WRITE CODE HERE
    y_pred = np.random.uniform(low=0.0, high=1.0, size=1000)
    y_true = np.random.randint(2, size=1000)
    k_range = np.arange (0, 1, 0.05)
    for k in k_range:
        y_list = np.copy(y_pred)
        y_list[y_list >= k] = 1
        y_list[y_list < k] = 0
        fpr_tpr = compute_fpr_tpr(y_true, y_list)
        output['fpr_list'].append(fpr_tpr['fpr']) 
        output['tpr_list'].append(fpr_tpr['tpr']) 
        
        
    #plot ROC curve using tpr , fpr, k
    
    #plt.plot(output['fpr_list'], output['tpr_list'], label=k_range)
    #plt.ylabel('True Positive Rate')
    #plt.xlabel('False Positive Rate for dumb model')
    #plt.show()

    return output


def compute_fpr_tpr_smart_model():
    """
    Simulates a smart model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds (k)
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05
                 ...
                 output['fpr_list'][-1] corresponds to k=0.95

            Do the same for output['tpr_list']
    """
    output = {'fpr_list': [], 'tpr_list': []}

    # WRITE CODE HERE
    y_true = np.random.randint(2, size=1000)
    index_p = np.where(y_true == 1)
    index_n = np.where(y_true == 0)
    y_pred = np.ones(len(y_true))

    for i in index_p[0]:
        rand = np.random.uniform(low=0.4, high=1.0)
        y_pred[i] = rand

    for i in index_n[0]:
        rand_n = np.random.uniform(low=0, high=0.6)
        y_pred[i] = rand_n

    k_range = np.arange (0, 1, 0.05)
    for k in k_range:
        y_list = np.copy(y_pred)
        y_list[y_list >= k] = 1
        y_list[y_list < k] = 0
        fpr_tpr = compute_fpr_tpr(y_true, y_list)
        output['fpr_list'].append(fpr_tpr['fpr']) 
        output['tpr_list'].append(fpr_tpr['tpr']) 

    #plot ROC curve using tpr , fpr, k
    
    #plt.plot(output['fpr_list'], output['tpr_list'], label=k_range)
    #plt.ylabel('True Positive Rate')
    #plt.xlabel('False Positive Rate for Smart Model')
    #plt.show()
    
    return output


def compute_auc_both_models():
    """
    Simulates a dumb model and a smart model and computes the AUC of both

    :Return: dict with auc_dumb_model, auc_smart_model.
             These contain the AUC for both models
             auc values in the lists should be floats
    """
    output = {'auc_dumb_model': 0., 'auc_smart_model': 0.}

    # WRITE CODE HERE
    
    y_true = np.random.randint(2, size=1000)
    
    y_pred_smart = np.ones(len(y_true))
    index_p = np.where(y_true == 1)
    index_n = np.where(y_true == 0)
    for i in index_p[0]:
        rand = np.random.uniform(low=0.4, high=1.0)
        y_pred_smart[i] = rand
    for i in index_n[0]:
        rand_n = np.random.uniform(low=0, high=0.6)
        y_pred_smart[i] = rand_n
 
    y_pred_dumb = np.random.uniform(low=0.0, high=1.0, size=1000) 
    
    
    auc_dumb = compute_auc(y_true, y_pred_dumb)
    auc_smart = compute_auc(y_true, y_pred_smart)
    
    output['auc_dumb_model'] = auc_dumb['auc']
    output['auc_smart_model'] = auc_smart['auc']

    return output


def compute_auc_untrained_model(model, dataloader, device):
    """
    Computes the AUC of your input model

    Dont forget to re-apply your output activation!

    :Return: dict with auc_dumb_model, auc_smart_model.
             These contain the AUC for both models
             auc values should be floats

    Make sure this function works with arbitrarily small dataset sizes!
    """
    output = {'auc': 0., 'fpr': [], 'tpr':[]}

    # WRITE CODE HERE
    y_pred = torch.from_numpy(np.array([])).to(device)
    y_true = torch.from_numpy(np.array([])).to(device)
    model.eval()
       
    for batch in dataloader:
        seq = batch['sequence'].to(device)
        
        
        y_model = model(seq)
        x = torch.sigmoid(y_model).detach().view(-1)
        y_pred = torch.cat([y_pred,x])
        target = batch['target'].to(device)
        x = target.detach().view(-1)
        y_true = torch.cat([y_true, x])
        
    model_auc = compute_auc(y_true.to('cpu').numpy() ,y_pred.to('cpu').numpy())
    output['auc'] = model_auc['auc']
    output['fpr'] = model_auc['fpr']
    output['tpr'] = model_auc['tpr']
    return output


def compute_auc(y_true, y_model):
    """
    Computes area under the ROC curve
    auc returned should be float
    Args:
        :param y_true: groundtruth labels (np.array of ints [0 or 1])
        :param y_model: model outputs (np.array of float32 in [0, 1])

    Note: if you set y_model as the output of solution.Basset, 
    you need to transform it first!
    """
    output = {'auc': 0., 'fpr':[], 'tpr': []}
    # WRITE CODE HERE
    
    fpr = []
    tpr = []
    threshold = list(range(0,100,5))
    for j , threshold in enumerate(threshold):
        threshold1 = threshold/100
        temp_pred = np.where(y_model>threshold1, 1, 0)
        result = compute_fpr_tpr(np.asarray(y_true), np.asarray(temp_pred))
        fpr.append(result["fpr"])
        tpr.append(result["tpr"])
    fpr1 = fpr[::-1]
    tpr1 = tpr[::-1]
    
    AUC = 0
    for i in range(1,len(fpr1)):
        width = (fpr1[i] - fpr1[i-1])
        left_area = (width * tpr1[i-1])
        right_area = (width * tpr1[i])
        ave = (left_area + right_area)/2
        AUC += ave

    output['auc'] = AUC
    output['fpr'] = fpr
    output['tpr'] = tpr
    return output


# In[ ]:


def get_critereon():
    """
    Picks the appropriate loss function for our task
    criterion should be subclass of torch.nn
    """
    
    # WRITE CODE HERE
    
    critereon = nn.BCEWithLogitsLoss()
    
    return critereon


def train_loop(model, train_dataloader, device, optimizer, criterion):
    """
    One Iteration across the training set
    Args:
        :param model: solution.Basset()
        :param train_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to display losses and/or scores within the loop, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!

    Note: you don’t need to compute the score after each training iteration.
    If you do this, your training loop will be really slow!
    You should instead compute it every 50 or so iterations and aggregate ...
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE
    model.train()
    y_list = []
    y_pred = []
    counter = 0
    store_every = 50
    train_loss = 0.0
    auc_running = 0.0
    #https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    for data in train_dataloader:
        counter += 1
        optimizer.zero_grad()
        x = data['sequence'].to(device)
        y = data['target'].to(device)
        outputs = model(x)
        
        loss = criterion(outputs,y)
        
        train_loss += loss
        print('counter is:', counter, 'train_loss is:', train_loss)
        loss.backward()
        optimizer.step()
        
        outputs = torch.sigmoid(outputs)
        
        y_list.append(y.cpu().numpy())
        y_pred.append(outputs.cpu().detach().numpy())
    
    y_list = np.concatenate(y_list, axis = 0)
    y_pred = np.concatenate(y_pred, axis = 0)
    
    target = y_list.reshape(-1)
    pred = y_pred.reshape(-1)
    
    model_auc = compute_auc(np.asarray(target), np.asarray(pred))
    
    print('model auc is:', model_auc['auc'])
    output['total_loss'] = train_loss
    output['total_score'] = model_auc['auc']

    return output['total_score'], output['total_loss']


def valid_loop(model, valid_dataloader, device, optimizer, criterion):
    """
    One Iteration across the validation set
    Args:
        :param model: solution.Basset()
        :param valid_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to display losses and/or scores within the loop, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE
    print('Validating')
    model.train()
    y_list = []
    y_pred = []
    counter = 0
    store_every = 50
    train_loss = 0.0
    auc_running = 0.0
    #https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    for data in valid_dataloader:
        counter += 1
        optimizer.zero_grad()
        x = data['sequence'].to(device)
        y = data['target'].to(device)
        outputs = model(x)
        
        loss = criterion(outputs,y)
        
        train_loss += loss
        print('validation counter is:', counter, 'valid_loss is:', train_loss)
        loss.backward()
        optimizer.step()
        
        outputs = torch.sigmoid(outputs)
        y_list.append(y.cpu().numpy())
        y_pred.append(outputs.cpu().detach().numpy())
    
    y_list = np.concatenate(y_list, axis = 0)
    y_pred = np.concatenate(y_pred, axis = 0)
    
    target = y_list.reshape(-1)
    pred = y_pred.reshape(-1)
    
    model_auc = compute_auc(np.asarray(target), np.asarray(pred))
    
    print('validation auc is', model_auc['auc'])
    output['total_loss'] = train_loss
    output['total_score'] = model_auc['auc']

    return output['total_score'], output['total_loss']

