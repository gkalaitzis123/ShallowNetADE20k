from ultralytics import YOLO
from PIL import Image
import os
import json
import pandas as pd

def bbIOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0]) 
    yA = max(boxA[1], boxB[1]) 
    xB = min(boxA[2], boxB[2]) 
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = int(max(0, xB - xA) * max(0, yB - yA))
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = int((boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
    boxBArea = int((boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

resultStorer = []
data = pd.read_csv('jsondata.csv')
model = YOLO('runs/detect/train5/weights/best.pt')
count = 0

for file in os.listdir("data/images/test"):
        
    # Read the image and perform object detection on it
    predictions = model(f"data/images/test/{file}", save_txt=None)
    prediction = predictions[0]
    YOLOPREDICTIONS = prediction.boxes  # Boxes object for bbox outputs
    specific_value = file
    GROUNDTRUTHS = data[data['0'] == specific_value]
    matrix = []
    gtDict = {}
    gtclassDict = {}
    
    if count == 1:
    
      break
    
    count += 1
    
    for YOLOPREDICTIONforimg in range(len(YOLOPREDICTIONS)):

        yoloTopLeftx = float(prediction.boxes[YOLOPREDICTIONforimg].xyxy[0][0]) 
        yoloTopLefty = float(prediction.boxes[YOLOPREDICTIONforimg].xyxy[0][1])
        yoloBottomRightx = float(prediction.boxes[YOLOPREDICTIONforimg].xyxy[0][2])
        yoloBottomRighty = float(prediction.boxes[YOLOPREDICTIONforimg].xyxy[0][3])
        yolobox = [yoloTopLeftx,yoloTopLefty,yoloBottomRightx,yoloBottomRighty]
        matrixrow = []

        for GROUNDTRUTH in range(len(GROUNDTRUTHS)):
            
            gtClass = GROUNDTRUTHS.iloc[GROUNDTRUTH][1]
            gtTopLeftx = GROUNDTRUTHS.iloc[GROUNDTRUTH][2]
            gtTopLefty = GROUNDTRUTHS.iloc[GROUNDTRUTH][5]
            gtBottomRightx = GROUNDTRUTHS.iloc[GROUNDTRUTH][4]
            gtBottomRighty = GROUNDTRUTHS.iloc[GROUNDTRUTH][3]
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
          #filename,class # predicted,iou of best match to ground truths, prob that model is correct,(of prediction)topleftx,toplefty,bottomrightx,bottomrighty,(of groundtruth)topleftx,toplefty,bottomrightx,bottomrighty
          newrow = [file,int(prediction.boxes.cls[rowi]),max(matrix[rowi]),float(prediction.boxes[rowi].conf),int(prediction.boxes[rowi].xyxy[0][0]),int(prediction.boxes[rowi].xyxy[0][1]),int(prediction.boxes[rowi].xyxy[0][2]),int(prediction.boxes[rowi].xyxy[0][3]),gtclassDict[rowi],gtDict[rowi][0],gtDict[rowi][1],gtDict[rowi][2],gtDict[rowi][3]]
          resultStorer.append(newrow)
    
#df = pd.DataFrame(resultStorer)
#df.to_csv('ResultsYoloADE.csv', index=False)