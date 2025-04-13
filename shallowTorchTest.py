import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
from sklearn.utils import resample

# possible prunings:
# prune.random_unstructured(model.fc1,name="weight",amount=.3)
# prune.ln_structured(model.fc1,name="weight",amount=.5,n=2,dim=1)
# prune.ln_unstructured(model.fc1,name="weight",amount=.5)

device = torch.device("cpu")


# pruning guidance from: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html

class MyNet(nn.Module):
    def __init__(self,numInput=94):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(numInput,85)

    def forward(self, x):
        x = self.fc1(x)
        return x

model=1
criterion=2
optimizer=3
#rebalanceBool=False

def initLearning(learnRate=1, reweight=False, rebalance=False, numInput=94):
  global model
  model = MyNet(numInput).to(device=device)
  
  global criterion
  
  criterion = nn.CrossEntropyLoss()
  
  global rebalanceBool
  rebalanceBool = rebalance
  
  global optimizer
  optimizer = torch.optim.SGD(model.parameters(), lr = learnRate) #lr=1 was good unweighted # lr was 0.01
  
  print('model and optimizer initialized')

def upsample(feats,labs):
   #classCount=df_LS['OBJECT (PAPER)'].value_counts()
   classCount=np.histogram(labs,100)[0]#np.unique(labs)))[0]
   maxClass=classCount.argmax()
   boostNum=np.zeros(classCount.shape[0])
   subClassDf={}
   subClassDf_upsamp={}
   featsList=feats.T.tolist()
   featsList.append(labs.tolist())
   feats=np.array(featsList).T
   for keyNum in range(len(classCount)):
      #boostNum[key]=classCount['person']/classCount[key]
      if classCount[keyNum] > 0:
        boostNum[keyNum]=classCount[maxClass]/classCount[keyNum]
        subClassDf[keyNum]=feats[np.where(labs==keyNum)[0],:]
        print(np.shape(subClassDf[keyNum]))
        subClassDf_upsamp[keyNum]=resample(subClassDf[keyNum],n_samples=int(np.round(classCount[keyNum]*boostNum[keyNum])),replace=True)
        continue

   allUpSamp=subClassDf_upsamp[0]
   for keyNum in range(1, len(classCount)):
      if classCount[keyNum] > 0:
        allUpSamp=np.hstack((allUpSamp.T,subClassDf_upsamp[keyNum].T)).T
   
   return allUpSamp[:,:-1], allUpSamp[:,-1]


def fit(net,feats,labs,batch_size=np.Inf, epochs=20, shuffle=True, valRat=.8,patience=5, lambda1=0.001, mustPrune=False):

    if shuffle:
        permVals=np.random.permutation(feats.shape[0])
        feats=feats[permVals,:]
        labs=labs[permVals]

    loss_hist=[]
    lossVal_hist=[]

    valPartit=int(np.floor(valRat*feats.shape[0]))
    featsTrain=feats[:valPartit,:]
    labsTrain=labs[:valPartit]
    featsVal=feats[valPartit:,:]
    labsVal=labs[valPartit:]
    
    #if rebalanceBool: # up-sample under-represented data
      #featsTrain,labsTrain = upsample(featsTrain, labsTrain)
      #featsVal, labsVal    = upsample(featsVal,   labsVal)

    dataFull=featsTrain
    labelsFull=labsTrain

    if batch_size<feats.shape[0]:
        totalBatches=np.floor(dataFull.shape[0]/batch_size)
        batchInd=0
    for e in range(epochs):
        train_loss = 0.0
        #for data, labels in tqdm(trainloader):
        # Clear the gradients
        optimizer.zero_grad()
        if batch_size>=feats.shape[0]:
            data=dataFull
            labels=labelsFull
        else:
            data=dataFull[batchInd*batch_size:(batchInd*(batch_size+1)-1),:]
            labels=labelsFull[batchInd*batch_size:(batchInd*(batch_size+1)-1)]
        # Forward Pass
        target = model(torch.Tensor(data))
        targetVal = model(torch.Tensor(featsVal))
        # Find the Loss
        allFC_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
        l1_loss=lambda1*torch.norm(allFC_params,1)
        loss = criterion(target,torch.LongTensor(labels)) + l1_loss
        lossVal = criterion(targetVal,torch.LongTensor(labsVal))
        loss_hist.append(loss.item())
        lossVal_hist.append(lossVal.item())
        if len(lossVal_hist)>patience:
            if np.min(lossVal_hist[-patience:])>lossVal_hist[-(patience+1)]:
                return loss_hist, lossVal_hist
        # Calculate gradients 
        loss.backward()
        # Update Weights
        optimizer.step()
        #test step
        #nRows = model.fc1.weight.data.shape[0]
        #identity_matrix = np.eye(nRows, dtype=np.float32)  # Create an identity matrix with the same type as the PyTorch tensor
        #model.fc1.weight.data[:nRows, :nRows] = torch.from_numpy(identity_matrix)
        # Add a pruning section
        if mustPrune:
            prune.ln_unstructured(model.fc1,name="weight",amount=.5,n=2,dim=1)
        # Calculate Loss
        train_loss += loss.item()
        if batch_size<feats.shape[0]:
            batchInd+=1
            if batchInd==totalBatches:
                batchInd=0
    return loss_hist, lossVal_hist

def fwdPass(inTensor):
  return model(inTensor)

def get_model() :
  return model
