#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
import torch.nn.functional as F


# In[ ]:


class SingleConvClassifier(nn.Module):
    def __init__(self, number_of_classes):
        super(SingleConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=4, stride=2, padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1)
        
        self.batch1 = torch.nn.BatchNorm2d(8, eps=0.001, momentum=0.99)
        self.batch2 = torch.nn.BatchNorm2d(16, eps=0.001, momentum=0.99)
        self.batch3 = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.99)
        self.batch4 = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.99)
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=1344, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=number_of_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x, kernel_size=2)
        
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x, kernel_size=2)
        
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x, kernel_size=2)
    
        x = self.conv4(x)
        x = self.batch4(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x, kernel_size=2)  
        
        x = torch.flatten(x, 1) 
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        
        output = F.log_softmax(x, dim=1)
        
        return output


# In[ ]:


class MultiConvClassifier(nn.Module):
    def __init__(self, number_of_classes):
        super(MultiConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=4, stride=2, padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1)
        
        self.batch1 = torch.nn.BatchNorm2d(8, eps=0.001, momentum=0.99)
        self.batch2 = torch.nn.BatchNorm2d(16, eps=0.001, momentum=0.99)
        self.batch3 = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.99)
        self.batch4 = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.99)
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
       
        self.fc1 = nn.Linear(in_features=1344, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=number_of_classes)
        
        self.fc3 = nn.Linear(in_features=number_of_classes, out_features=number_of_classes)
        self.fc4 = nn.Linear(in_features=number_of_classes, out_features=number_of_classes)
        self.fc5 = nn.Linear(in_features=number_of_classes, out_features=number_of_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x, kernel_size=2)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x, kernel_size=2)

        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x, kernel_size=2)

        x = self.conv4(x)
        x = self.batch4(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x, kernel_size=2)

        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        
        output = self.fc5(x)

        return output

