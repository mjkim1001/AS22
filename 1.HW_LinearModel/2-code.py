#!/usr/bin/env python
# coding: utf-8

# In[50]:


# Don't change batch size, but you can change where to save the data
batch_size = 64
data_dir = './data'

from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd

## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA

train_data = datasets.MNIST(data_dir, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

# Once you have downloaded the data by setting download=True, you can
# change download=True to download=False
test_data = datasets.MNIST(data_dir, train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

subset_indices = ((train_data.targets == 0) + (train_data.targets == 1)).nonzero() #nonzero returns indexes, otherwise T/F result.
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, 
  shuffle=False,sampler=SubsetRandomSampler(subset_indices.view(-1)))


subset_indices = ((test_data.targets == 0) + (test_data.targets == 1)).nonzero()
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, 
  shuffle=False,sampler=SubsetRandomSampler(subset_indices.view(-1)))


# In[51]:


input_dim = 28*28
output_class = 1
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_class):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_class)
        
    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    
class Logistic_Loss(nn.modules.Module):
    def __init__(self):
        super(Logistic_Loss, self).__init__()
    def forward(self, outputs, labels):
        batch_size = outputs.size()[0]
        return torch.sum(torch.log(1+ torch.exp(-(outputs.t()*labels))))/batch_size

class SVM_Loss(nn.modules.Module):
    def __init__(self):
        super(SVM_Loss, self).__init__()
    def forward(self, outputs, labels):
        return torch.sum( torch.clamp(1- outputs.t()*labels, min=0)/batch_size )
    
    
model_lr = LogisticRegression(input_dim, output_class)
loss_lr = Logistic_Loss()


# In[52]:



class SVM_Loss(nn.modules.Module):
    def __init__(self):
        super(SVM_Loss, self).__init__()
    def forward(self, outputs, labels):
        return torch.sum( torch.clamp(1- outputs.t()*labels, min=0)/batch_size )

model_svm = nn.Linear(input_dim, output_class)
loss_svm = SVM_Loss()


# In[72]:


num_epochs = 10
def run_sgd(model, loss, test_fun, num_epochs, learning_rate, momentum, text):
    sgd_optimizer= torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
    print ('Learning rate : {}, momentum : {:.1f}' .format(learning_rate, momentum))
    
    listofloss=[]
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0 

        for i, (images, labels) in enumerate(train_loader):
            #train loader reads train dataset for the batch size.
            # Convert the 28*28 image matrix into a 784-dim vector
            images = images.view(-1, 28*28) 
            # Convert labels from 0,1 to -1,1
            labels = 2*(labels.float()-0.5)

            ## TODO 
            # 1. Compute Loss. Check torch functions for the corresponding loss for Logistic and SVM
            outputs = model(images)
            loss_per_batch = loss(outputs,Variable(labels))

            # 2. Do optimization. Chec torch.optim to see how to do optimization with pytorch
            ##################SGD#####################
            sgd_optimizer.zero_grad()
            loss_per_batch.backward()
            sgd_optimizer.step()

            #sgd_optimizer_svm.zero_grad()
            # 3. Save batch loss
            batch_count += 1
            total_loss += loss_per_batch.item()

        ## Print your results every epoch
        avg_loss = total_loss/batch_count
        listofloss.append(avg_loss)
        print ('Epoch [{}/{}], {}:[average loss {:.4f}]' 
                       .format(epoch+1, num_epochs, text, avg_loss))
    # Test the Model
    correct = 0.
    total = 0.
    for images, labels in test_loader:
        images = images.view(-1, 28*28)

        ## Replace your prediction code here, currently it's a random prediction
        predict_test = test_fun(model(images))
        prediction = predict_test.data >= 0

        #prediction_svm = model_svm(images).data >= 0

        correct += (prediction.view(-1).long() == labels).sum()
        #correct_svm += (prediction_svm.view(-1).long() == labels).sum()
        #.long() converts T/F into 0,1
        total += images.shape[0]

    print('Accuracy on the test images: %f %%' % (100 * (correct.float() / total)))
    testerr = (100 * (correct.float() / total))
    return listofloss
        
learning_rate = 0.01
momentum = 0.0
def act_lr(model):
    return torch.sigmoid(model)-0.5
def act_svm(model):
    return model

#run_sgd(model_lr, loss_lr, act_lr, num_epochs, learning_rate, momentum, "Logistic Regression SGD")
#run_sgd(model_svm, loss_svm, act_svm, num_epochs, learning_rate, momentum,"Linear SVM SGD")



# In[73]:


torch.manual_seed(2022)
learning_rate = 0.01
model_lr = LogisticRegression(input_dim, output_class)
loss_lr = Logistic_Loss()
model_svm = nn.Linear(input_dim, output_class)
loss_svm = SVM_Loss()
df = pd.DataFrame()
df["Logistic"] = run_sgd(model_lr, loss_lr, act_lr, num_epochs, learning_rate, momentum, "Logistic Regression SGD")
df["SVM"] = run_sgd(model_svm, loss_svm, act_svm, num_epochs, learning_rate, momentum,"Linear SVM SGD")

df.plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(title='model')
plt.show()


# In[114]:


df_m = pd.DataFrame()
learning_rate = 0.01
for momentum in np.arange(0.0,1.0,0.2):
    torch.manual_seed(2022)
    model_lr = LogisticRegression(input_dim, output_class)
    loss_lr = Logistic_Loss()
    df_m[momentum]= run_sgd(model_lr, loss_lr, act_lr, num_epochs, learning_rate, momentum, "Logistic Regression SGD")
df_m.plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(title='momentum')
plt.show()


# In[115]:


df_m_svm = pd.DataFrame()
learning_rate = 0.01
for momentum in np.arange(0.0,1.0,0.2):
    torch.manual_seed(2022)
    model_svm = nn.Linear(input_dim, output_class)
    loss_svm = SVM_Loss()
    df_m_svm[momentum] = run_sgd(model_svm, loss_svm, act_svm, num_epochs, learning_rate, momentum,"Linear SVM SGD")
df_m_svm.plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(title='momentum')
plt.show()


# In[87]:


df = pd.DataFrame()
momentum = 0.0
for learning_rate in [1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, 0.1]:
    torch.manual_seed(2022)
    model_lr = LogisticRegression(input_dim, output_class)
    loss_lr = Logistic_Loss()
    df[learning_rate] = run_sgd(model_lr, loss_lr, act_lr, num_epochs, learning_rate, momentum, "Logistic Regression SGD")
df.plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(title='learning rate')
plt.show()


# In[121]:


df = pd.DataFrame()
momentum = 0.8
for learning_rate in [1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, 0.1]:
    torch.manual_seed(2022)
    model_lr = LogisticRegression(input_dim, output_class)
    loss_lr = Logistic_Loss()
    df[learning_rate]= run_sgd(model_lr, loss_lr, act_lr, num_epochs, learning_rate, momentum, "Logistic Regression SGD")
df.plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(title='learning rate')
plt.show()


# In[110]:


df_svm = pd.DataFrame()
momentum = 0.0
for learning_rate in [1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, 0.1]:
    torch.manual_seed(2022)
    model_svm = nn.Linear(input_dim, output_class)
    loss_svm = SVM_Loss()
    df_svm[learning_rate] = run_sgd(model_svm, loss_svm, act_svm, num_epochs, learning_rate, momentum,"Linear SVM SGD")
df_svm.plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(title='learning rate')
plt.show()


# In[120]:


df_svm = pd.DataFrame()
momentum = 0.8
for learning_rate in [1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, 0.1]:
    torch.manual_seed(2022)
    model_svm = nn.Linear(input_dim, output_class)
    loss_svm = SVM_Loss()
    df_svm[learning_rate] = run_sgd(model_svm, loss_svm, act_svm, num_epochs, learning_rate, momentum,"Linear SVM SGD")
df_svm.plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(title='learning rate')
plt.show()


# In[ ]:




