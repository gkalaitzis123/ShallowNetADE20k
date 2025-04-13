from ultralytics import YOLO
import os
import pandas as pd
import numpy as np

def bbIOU(boxA, boxB):
    xA = max(boxA[0], boxB[0]) 
    yA = max(boxA[1], boxB[1]) 
    xB = min(boxA[2], boxB[2]) 
    yB = min(boxA[3], boxB[3])
    interArea = int(max(0, xB - xA) * max(0, yB - yA))
    boxAArea = int((boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
    boxBArea = int((boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

resultStorer = []
objects = ['person, individual, someone, somebody, mortal, soul','tree','door','chair','car, auto, automobile, machine, motorcar','plant, flora, plant life','painting, picture','table','cabinet','lamp','cushion','signboard, sign','curtain, drape, drapery, mantle, pall','book','mirror','bed','vase','pillow','plate','rug, carpet, carpeting','sink', 'bench','towel','sconce','plaything, toy','glass, drinking glass','desk','animal, animate being, beast, brute, creature, fauna','computer, computing machine, computing device, data processor, electronic computer, information processing system','drawer','rock, stone','river','bottle','grass','sand','bathtub, bathing tub, bath, tub','toilet, can, commode, crapper, pot, potty, stool, throne']
data = pd.read_csv('C://Users//gkrul//OneDrive//Desktop//LLMkit//jsondataLLM.csv')
model = YOLO('C://Users//gkrul//OneDrive//Desktop//LLMkit//LLMtestYOLO20Epoch.pt')
for file in os.listdir("C://Users//gkrul//OneDrive//Desktop//LLMkit//yoloDataLLM//images//test"):
    
    try:
      predictions = model(f"C://Users//gkrul//OneDrive//Desktop//LLMkit//yoloDataLLM//images//test//{file}", save_txt=None)
    except IndexError:
      os.remove(f"C://Users//gkrul//OneDrive//Desktop//LLMkit//yoloDataLLM//images//test//{file}")
      continue
    prediction = predictions[0]
    YOLOPREDICTIONS = prediction.boxes  # Boxes object for bbox outputs
    specific_value = file
    GROUNDTRUTHS = data[data['0'] == specific_value]
    if GROUNDTRUTHS.shape[0] > 0:
      scene = GROUNDTRUTHS.iat[0,1]
    matrix = []
    gtDict = {}
    gtclassDict = {}

    for YOLOPREDICTIONforimg in range(len(YOLOPREDICTIONS)):

        yoloTopLeftx = float(prediction.boxes[YOLOPREDICTIONforimg].xyxy[0][0]) 
        yoloTopLefty = float(prediction.boxes[YOLOPREDICTIONforimg].xyxy[0][1])
        yoloBottomRightx = float(prediction.boxes[YOLOPREDICTIONforimg].xyxy[0][2])
        yoloBottomRighty = float(prediction.boxes[YOLOPREDICTIONforimg].xyxy[0][3])
        yolobox = [yoloTopLeftx,yoloTopLefty,yoloBottomRightx,yoloBottomRighty]
        matrixrow = []

        for GROUNDTRUTH in range(len(GROUNDTRUTHS)):
            
            gtClass = GROUNDTRUTHS.iloc[GROUNDTRUTH][2]
            gtTopLeftx = GROUNDTRUTHS.iloc[GROUNDTRUTH][3]
            gtTopLefty = GROUNDTRUTHS.iloc[GROUNDTRUTH][6]
            gtBottomRightx = GROUNDTRUTHS.iloc[GROUNDTRUTH][5]
            gtBottomRighty = GROUNDTRUTHS.iloc[GROUNDTRUTH][4]
            gtbox = [gtTopLeftx,gtTopLefty,gtBottomRightx,gtBottomRighty]
            
            if len(matrixrow) == 0:
              gtDict[YOLOPREDICTIONforimg] = gtbox
              gtclassDict[YOLOPREDICTIONforimg] = gtClass
            
            elif bbIOU(yolobox,gtbox)>max(matrixrow):
              gtDict[YOLOPREDICTIONforimg] = gtbox
              gtclassDict[YOLOPREDICTIONforimg] = gtClass
            
            matrixrow.append(bbIOU(yolobox,gtbox))
            
        matrix.append(matrixrow)

    for rowi in range(len(matrix)):
        
        if len(matrix[rowi]) > 0:
          #filename, predicted class, iou, predicted class confidence,
          #(of prediction)topleftx,toplefty,bottomrightx,bottomrighty,
          #(of groundtruth) real class, topleftx,toplefty,bottomrightx,bottomrighty, 
          #conf of all other classes
          newrow = [file,scene,objects[int(prediction.boxes.cls[rowi])],max(matrix[rowi]),float(prediction.boxes[rowi].conf),
                    int(prediction.boxes[rowi].xyxy[0][0]),int(prediction.boxes[rowi].xyxy[0][1]),int(prediction.boxes[rowi].xyxy[0][2]),int(prediction.boxes[rowi].xyxy[0][3])
                    ,gtclassDict[rowi],gtDict[rowi][0],gtDict[rowi][1],gtDict[rowi][2],gtDict[rowi][3]]
          
          if newrow[3] < 0.5: #iou check
             
             continue
          
          for clsConf in prediction.boxes.data[rowi][6:]:

            newrow.append(float(clsConf))

          resultStorer.append(newrow)

columns = ['fname', 'scene', 'maxClass', 'iouWithgt', 'maxConf', 'pred_x1', 'pred_y1', 'pred_x2', 'pred_y2', 'gtClass', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2'] + [f'class_confidence_{i}' for i in range(37)]
df = pd.DataFrame(resultStorer, columns=columns)
df.to_csv('YOLOresults20EpochLLM.csv', index=False)