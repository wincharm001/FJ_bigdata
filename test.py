from ultralytics import YOLO

model = YOLO("weight\yolov8n-seg.pt")

model.train(data="config.yaml", epochs=100, imgsz=640)
