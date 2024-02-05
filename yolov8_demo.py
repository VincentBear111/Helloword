import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.yaml")        # 从头开始构建新模型
model = YOLO("yolov8n.pt")          # 加载预训练模型（建议用于训练）

# 使用模型
model.train(data="coco128.yaml", epochs=3, workers=0)  # 训练模型
metrics = model.val()  # 在验证集上评估模型性能
results = model("./bus.jpg")  # 对图像进行预测

print("--------------------------------------")
print(results)
# success = model.export(format="onnx")  # 将模型导出为 ONNX 格式