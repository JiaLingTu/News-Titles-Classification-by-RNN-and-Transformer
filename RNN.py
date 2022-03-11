# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:06:41 2021

@author: JiaLing Tu
"""
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
from collections import Counter
from torchtext.vocab import Vocab
from torch.autograd import Variable
from model import RNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn1(batch): # Train
    text_list, label_list = [], []
    for (_, label_, text_) in batch:
        label_list.append(label_pipeline(label_))
        process_text = torch.tensor(text_pipeline(text_), dtype=torch.int64)
        text_list.append(process_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return label_list.to(device), text_list.to(device)

def collate_fn2(batch):# Train
    text_list= []
    for (_, text_) in batch:
        process_text = torch.tensor(text_pipeline(text_), dtype=torch.int64)
        text_list.append(process_text)
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return text_list.to(device)

def count_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters

# load data from csv
train_df = pd.read_csv(r'.\data\train.csv')
train_result =list(train_df.to_records(index=False)) # list of tuples
test_df = pd.read_csv(r'.\data\test.csv')
test_result =list(test_df.to_records(index=False)) # list of tuples


# extract vocab from pre-trained embedding vocab
tokenizer = RegexpTokenizer(r'\w+')
# Find words
ps = PorterStemmer()
counter = Counter()
stop_words = set(stopwords.words('english')) 

l = []
for title in train_df['Title']:
    tokens = tokenizer.tokenize(title)
    new_tokens = [ps.stem(x) for x in tokens]
    new_tokens = [w for w in new_tokens if not w in stop_words]
    counter.update(new_tokens)
    l.append(new_tokens)
vocab = Vocab(counter, min_freq=2, vectors='glove.6B.300d')
embed = vocab.vectors

# ==================================================================================================
# hyper-parameter
model = RNN(embed, hidden_dim1=64, n_class=5, dropout = 0.5)
lr = 0.001
epoch = 15
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[3,6,10])


# ==================================================================================================

num_train, num_val = len(train_result) * np.array([0.8, 0.2])
train_set, val_set = torch.utils.data.random_split(train_result, [int(num_train), int(num_val)], generator=torch.Generator().manual_seed(0))

# define some function
label2num_Dict = {"sport":0, "entertainment":1, "politics":2, "business":3, "tech":4}
label_pipeline = lambda item: label2num_Dict[item]
text_pipeline = lambda x: [vocab[ps.stem(token)] for token in tokenizer.tokenize(x) if not token in stop_words]

# batch up training data 
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn1)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, collate_fn=collate_fn1)

loss_fn = nn.CrossEntropyLoss()
epoch_train_loss = []
epoch_train_acc = []
epoch_val_loss = []
epoch_val_acc = []

for epoch_idx in range(1, epoch+1):
    # print(scheduler.get_last_lr())
    print('==== {}th epoch ===='.format(epoch_idx))
    epoch_start = time.time()
    accurate_count = 0
    batch_loss = 0
    
    model.train()
    for batch_idx, (label, text) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        
        # forward
        out = model(text)
        
        # cal loss
        loss = loss_fn(out, label)
        batch_loss += loss
        
        # Backpropagation (BP)
        loss.backward()
        optimizer.step()
 
        # calculate accuracy
        _, predicted = torch.max(out, 1)
        tmp = np.count_nonzero((predicted==label).cpu().detach().numpy())
        accurate_count += tmp
        # End of Train    
    epoch_train_loss.append(batch_loss.detach().numpy()/(batch_idx+1))
    epoch_train_acc.append(round(accurate_count/num_train, 7)*100)    

    # validation
    val_loss = 0
    val_acc_count = 0
    model.eval()        
    with torch.no_grad():
        for batch_idx, (label, text) in enumerate(tqdm(val_loader)):
            # forward
            out = model(text)
            
            # cal loss
            loss = loss_fn(out, label)
            val_loss += loss
                        
            # calculate accuracy
            _, predicted = torch.max(out, 1)
            valtmp = np.count_nonzero((predicted==label).cpu().detach().numpy())
            val_acc_count += valtmp    
            
    scheduler.step()
    epoch_val_loss.append(val_loss.detach().numpy()/(batch_idx+1))
    epoch_val_acc.append(round(val_acc_count/num_val, 7)*100)       
    print('================================')
    print('training loss: {}'.format(epoch_train_loss[-1]))
    print('training acc: {}%'.format(epoch_train_acc[-1]))
    print('validation loss: {}'.format((epoch_val_loss[-1])))
    print('validation acc: {}%'.format(epoch_val_acc[-1]))
    
    per_epoch_time = time.time() - epoch_start
    print('train and test cost {} seconds'.format(per_epoch_time))
print('\nFin.')

#%%
# Batch up test data
test_loader = torch.utils.data.DataLoader(test_result, batch_size=1, shuffle=False, collate_fn=collate_fn2)   

test_output = [] 
with torch.no_grad():
    for input_tensor in tqdm(test_loader):
        # forward
        out = model(input_tensor)
        _, predicted = torch.max(out, 1)
        test_output.append(int(predicted.cpu().detach().numpy()))

num2label = {0:"sport", 1:"entertainment", 2:"politics", 3:"business", 4:"tech"}
relabel = lambda item: num2label[item]
relabel_output = list(map(relabel, test_output))

# dataframe to csv
submission = {'Id': np.arange(0,len(relabel_output)), 'Category': relabel_output}
df = pd.DataFrame(submission, columns= ['Id','Category'])
df.to_csv (r'.\output_RNN.csv', index=False, header=True)   
#%%
plt.figure(2)
plt.subplot(121)
plt.title("Average Loss")
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.plot(epoch_train_loss, 'b', label = 'Training')
plt.plot(epoch_val_loss, 'r', label = 'Validation')
plt.legend(loc=0)


plt.subplot(122)
plt.title("Total Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(epoch_train_acc, 'b', label = 'Train')
plt.plot(epoch_val_acc, 'r', label = 'Validation')
plt.legend(loc=4)
