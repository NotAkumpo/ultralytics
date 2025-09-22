from ultralytics import YOLO
import torch
torch.cuda.empty_cache()

if __name__ == '__main__':
    # model_path = "ultralytics\\cfg\\models\\v8\\yolov8nDilated.yaml"
    model_path = "dilatedRuns888\\train32\\weights\\last.pt"
    model = YOLO(model_path)

    # dataset = "C:\\Users\\Abdiel\\Documents\\GitHub\\ComputerVision\\ComputerVision\\YOLO\\CocoVehiclesDataset\\data.yaml"
    dataset = "C:\\Users\\Abdiel\\Documents\\GitHub\\ComputerVision\\ComputerVision\\YOLO\\VehiclesA-1\\data.yaml"

    results = model.train(data=dataset, epochs=100, resume=True, imgsz=640, device=0, project='dilatedRuns888')

    metrics = model.val(data=dataset, device=0)
 
    test_results = model.val(data=dataset, split='test', device=0) # Adjust device as needed