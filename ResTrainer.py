import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
from torch.utils.data.sampler import SubsetRandomSampler
import h5py

data_dir = "resnetData//train"
test_dir = "resnetData//test"
classes = []

for directory, subdirectories, files in os.walk(data_dir):
    for file in files:
        if directory.split("\\")[-1] not in classes:
            classes.append(directory.split("\\")[-1])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)])        
   
numEpochs = 100 # int

def get_train_valid_loader(data_dir,block,augment=0,random_seed=69420,valid_size=0,shuffle=False,show_sample=False,num_workers=4, pin_memory=False, batch_size=128):

    train_dataset = torchvision.datasets.ImageFolder(root=data_dir,transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir,transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,)
    valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,)
    
    return (train_loader, valid_loader)

# criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    for epoch in range(int(numEpochs)):  # loop over the dataset multiple times
        prev_loss = 100000.0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if epoch % 5 == 4: # print every 5 epochs
            # 	print('[%d, %5d] loss: %.3f' %
            # 		(epoch + 1, i + 1, running_loss / 100))
            # if i % 10 == 9:    # check to break every 10 mini-batches
            # 	# end if loss goes up # < (0.1 * prev_loss / 100):
            # 	# end if loss decreases by less than 10%
            # 	# if (prev_loss / 100 - running_loss / 100) < 0: 
            # 	# 	return "Done"
            # 	prev_loss = running_loss
            # 	running_loss = 0.0
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/100))

allAllAccs = []
cv_acc_all = []
for j in range(1):
    net = torchvision.models.resnet18(pretrained=False)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    net = net.to(device)
    trainloader, validloader = get_train_valid_loader(data_dir,block=j,shuffle=False,num_workers=0,batch_size=128)
    train()
    torch.save(net, 'resNetAde20KScenes100epochV3')
    accs = []
    correct = 0
    total = 0
    t2, validloader2 = get_train_valid_loader(data_dir,block=j,shuffle=False,num_workers=0,batch_size=128)
    with torch.no_grad():
        for data in validloader2:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
    print('Accuracy: %f %%' % (acc))
    cv_acc_all.append(acc)