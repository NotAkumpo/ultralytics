from ultralytics import YOLO
import torch
from roboflow import Roboflow
torch.cuda.empty_cache()

if __name__ == '__main__':

    rf = Roboflow(api_key="znnkxI9VMR08aa8lCNlu")
    project = rf.workspace("thesis-dekry").project("vehiclesa")
    version = project.version(1)
    dataset = version.download("yolov8")
                
    model_path = "ultralytics\\cfg\\models\\v8\\yolov8nDilated.yaml"
    # model_path = "dilatedRuns888\\train32\\weights\\last.pt"
    model = YOLO(model_path)

    # dataset = "C:\\Users\\Abdiel\\Documents\\GitHub\\ComputerVision\\ComputerVision\\YOLO\\CocoVehiclesDataset\\data.yaml"
    # dataset = "C:\\Users\\Abdiel\\Documents\\GitHub\\ComputerVision\\ComputerVision\\YOLO\\VehiclesA-1\\data.yaml"

    results = model.train(data=f'{dataset.location}/data.yaml', epochs=100, resume=True, imgsz=640, device="cpu", project='dilatedRuns888')

    metrics = model.val(data=f'{dataset.location}/data.yaml', device="cpu")

    test_results = model.val(data=f'{dataset.location}/data.yaml', split='test', device="cpu") # Adjust device as needed