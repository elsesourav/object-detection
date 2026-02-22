from ultralytics import YOLO
model = YOLO("yolo12m.pt")
model.export(format="onnx")