from ultralytics import YOLO

model_path = "ultralytics\\cfg\\models\\v8\\yolov8mDilated.yaml"
model = YOLO(model_path)
# model = YOLO("yolov8n.pt")
print(model)
model.info()