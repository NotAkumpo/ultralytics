from ultralytics import YOLO

model_path = "C:\\Users\\abdie\\Documents\\GitHub\\ultralytics\\ultralytics\\cfg\\models\\v8\\yolov8m.yaml"
model = YOLO(model_path)
# model = YOLO("yolov8n.pt")
# print(model)
model.info()