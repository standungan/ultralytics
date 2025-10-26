from  ultralytics import YOLO
import torch 

# define model
# model_yaml = "yolo11n.yaml"
# model_yaml = "cbambifpnv2_yolo11n.yaml"
model_yaml = "cbambifpnv3_yolo11n.yaml"
model = YOLO(model_yaml)

x = torch.randn(1, 3, 640, 640)

with torch.no_grad():
    features = model.model(x)

for feat in features:
    print("feature shape:", feat.shape)
res = model(x)
print("boxes tensor shape:", res[0].boxes.xyxy.shape)
