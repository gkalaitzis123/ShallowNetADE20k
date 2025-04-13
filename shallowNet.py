from shallowTorchTest import *
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
import pickle

dfKnownOverlap=pd.read_csv('ResultsYoloADEioucap.csv')

def str2Num(inStr) :
    possibleVals = ['chair', 'car, auto, automobile, machine, motorcar', 'plant, flora, plant life', 'floor, flooring', 'sky', 'windowpane, window', 'painting, picture', 'ceiling', 'table', 'cabinet', 'lamp', 'cushion', 'signboard, sign', 'drawer', 'sidewalk, pavement', 'road, route', 'curtain, drape, drapery, mantle, pall', 'book', 'box', 'streetlight, street lamp', 'bottle', 'wheel', 'mountain, mount', 'earth, ground', 'grass', 'seat', 'pot, flowerpot', 'rock, stone', 'armchair', 'spotlight, spot', 'mirror', 'bed', 'flower', 'vase', 'fence, fencing', 'pillow', 'sofa, couch, lounge', 'column, pillar', 'glass, drinking glass', 'sconce', 'plate', 'pole', 'wall socket, wall plug, electric outlet, electrical outlet, outlet, electric receptacle', 'rug, carpet, carpeting', 'bowl', 'balcony', 'sink', 'railing, rail', 'work surface', 'bench', 'house', 'headlight', 'license plate', 'taillight', 'desk', 'bag', 'basket, handbasket', 'coffee table, cocktail table', 'palm, palm tree', 'towel', 'jar', 'stool', 'pot', 'swivel chair', 'traffic light, traffic signal, stoplight', 'plaything, toy', 'fluorescent, fluorescent fixture', 'sea', 'clock', 'awning, sunshade, sunblind', 'skyscraper', 'shrub, bush', 'candlestick, candle holder', 'apron', 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'flag', 'ball', 'windshield', 'figurine, statuette', 'chandelier, pendant, pendent', 'paper', 'television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box', 'van', 'field', 'switch, electric switch, electrical switch']
    for i in range(len(possibleVals)):
        if inStr==possibleVals[i]:
            return i

# generate indices to extract train and test data
def trainTestInds(dataSize,kSplits,numSplit):
   trainInd=list(range(dataSize))
   splitSize=int(np.round(dataSize/kSplits))
   testInd =list(range(splitSize*numSplit,splitSize*(numSplit+1)))
   for num in testInd:
     trainInd=np.delete(trainInd,np.where(trainInd==num)[0])
   return trainInd, testInd

dfScenes=dfKnownOverlap['fname']
wtsDict={}
confusDict={}
errDict={}
srcDict={}
accVec=np.zeros((10))
recallVec=np.zeros((10))

def saveResults(filename):
  outDict={}
  outDict['recallVec']=recallVec
  outDict['accVec']=accVec
  outDict['confusDict']=confusDict
  outDict['wtsDict']=wtsDict
  outDict['errDict']=errDict
  outDict['srcDict']=srcDict
  pickle.dump(outDict,open(filename,'wb'))
  print(filename+' written')

def runLearnAndEval(numPCs=9):

  dfPCA=pd.read_csv('Ade20kResNetResponseMulticlass.csv',index_col=0)
  
  indList=dfPCA.index
  
  indListNew=[]
  for locStr in indList:
    indListNew.append(locStr[-18:-4])
  dfPCA.index=indListNew

  numInput=85+numPCs
  xMat=np.zeros((dfKnownOverlap.shape[0],numInput))
  xMat[:,:85]=dfKnownOverlap.iloc[:,13:].to_numpy()
  srcVec=dfKnownOverlap.iloc[:,0]
  
  errorList=[]
  rInd=0
  for row in dfScenes:
    try:
      xMat[rInd,85:(85+numPCs)]=dfPCA[dfPCA.index==row[-18:-4]].to_numpy()[:,:numPCs]
    except:
      errorList.append(row)
    rInd+=1

  #need to train both models on same data, we have 2270 rows in the YOLO results that we cannot use with the current models
  
  trueLabs=np.zeros(dfKnownOverlap.shape[0])

  for i in range(trueLabs.shape[0]):
     trueLabs[i]=str2Num(dfKnownOverlap['gtClass'].iloc[i])
  
  permScenes=True
  # rand permute scenes
  if permScenes:
    picNums=dfKnownOverlap['fname'].to_numpy()
    picNumsUnique=np.unique(picNums)
    indPerm=np.random.permutation(picNumsUnique.shape[0])
    picNumsPerm=picNumsUnique[indPerm]
    xMatCopy=xMat.copy()
    srcVecCopy=srcVec.copy()
    trueLabsCopy=trueLabs.copy()
    k=0
    for sceneInd in range(len(picNumsPerm)):
      currPicRows=np.where(picNums==picNumsPerm[sceneInd])[0]
      xMatCopy[list(range(k,k+len(currPicRows))),:]=xMat[currPicRows,:]
      srcVecCopy[list(range(k,k+len(currPicRows)))]=srcVec[currPicRows]
      trueLabsCopy[list(range(k,k+len(currPicRows)))]=trueLabs[currPicRows]
      k=k+len(currPicRows)
    
    xMat=xMatCopy
    srcVec=srcVecCopy
    trueLabs=trueLabsCopy
  
  trueY = np.zeros((trueLabs.shape[0],85))
  
  for i in range(trueLabs.shape[0]):
     trueY[i,int(trueLabs[i])]=1
  
  # normalize data:
  xMatNorm=xMat.copy()
  for col in range(xMat.shape[1]):
    if np.std(xMat[:,col])>0:
      xMatNorm[:,col] = (xMat[:,col]-xMat[:,col].mean())/np.std(xMat[:,col])
    else:
      print('uhoh '+str(col))
  xMat=xMatNorm
  
  for fold in range(10):
    
    trainInd, testInd = trainTestInds(xMat.shape[0],10,fold)
    initLearning(learnRate=0.1,rebalance=True, numInput=94)

    print(np.shape(testInd))

    global model

    [histTrain,histVal]=fit(model,xMat[trainInd,:],trueLabs[trainInd],epochs=4000,shuffle=False,valRat=0.9,patience=60) #,mustPrune=True,smartInit=True)

    try:
      newResults=fwdPass(torch.Tensor(xMat[testInd,:])).detach().numpy()
    except:
      newResults=fwdPass(torch.Tensor(xMat[testInd[:-20],:])).detach().numpy()

    newLabels=np.argmax(newResults,axis=1)

    try:
      yLabels=np.argmax(trueY[testInd,:],axis=1)
    except:
      yLabels=np.argmax(trueY[testInd[:-20],:],axis=1)


    ## save accuracies and network weights on each fold
    accVec[fold]=accuracy_score(yLabels,newLabels)
    recallVec[fold]=recall_score(yLabels,newLabels,average='macro')
    wtsDict[fold]=get_model().fc1.weight.detach().numpy()
    confusDict[fold]=confusion_matrix(yLabels,newLabels)
    errDict[fold]=(newLabels*100)-yLabels
    try:
      srcDict[fold]=srcVec[testInd]
    except:
      srcDict[fold]=srcVec[testInd[:-20]]
    print(fold)
    print("fold completed")

    print(accVec[fold])
    print(recallVec[fold])

  saveResults('shallowTest.pkl')
  
runLearnAndEval(9)