from PIL import Image
import pandas as pd
import torchvision
from torchvision import models
import torch
import torch.nn as nn
import numpy as np
import os
import torchvision.transforms as transforms
import shutil

data_dir = "resnetData//train"
test_dir = "resnetData//test"

def get_train_valid_loader(data_dir,block,augment=0,random_seed=69420,valid_size=0,shuffle=False,show_sample=False,num_workers=4, pin_memory=False, batch_size=128):

    train_dataset = torchvision.datasets.ImageFolder(root=data_dir,transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir,transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,)
    valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,)
    
    return (train_loader, valid_loader)

transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(
	mean = [0.485, 0.456, 0.406],
	std = [0.229, 0.224, 0.225]
)])

#"resNetAde20KScenes100epochV2"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torch.load("resNetAde20KScenes100epochV2")
net.eval()
accs = []
correct = 0
total = 0
t2, validloader2 = get_train_valid_loader(data_dir,block=1,shuffle=False,num_workers=0,batch_size=128)

for data in validloader2:

    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted,labels)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    
print('Accuracy: %f %%' % (acc))

#for directory, subdirectories, files in os.walk(data_dir):
    
    #for file in files:
        
        #file_path = os.path.join(directory, file)
        #img = Image.open(file_path)
        #img_t = transform(img)
        #batch_t = torch.unsqueeze(img_t,0)
        #out = net(batch_t)
        #print((out == torch.max(out)).nonzero(as_tuple=True), torch.topk(out,5)[1], torch.max(out))
        
    #if count == 10:
    
      #break
      
    #count += 1
    