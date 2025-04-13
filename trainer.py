from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(data="ADE20k.yaml", epochs=20)