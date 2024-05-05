import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO


def main():

    # 使用预训练模型权重训练模型 -- 预训练模型为 Yolov8x.pt
    # Load a model
    model = YOLO('yolov8x-p2-att.yaml')  # build a new model from YAML
    model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8x-p2-att.yaml').load('yolov8x.pt')  # build from YAML and transfer weights

    # 不用训练就可以查看模型结构
    model.info()

    # Train the model
    results = model.train(data='I:\\ultralytics\\detasets\\Axis\\my_data.yaml', 
                          epochs=50, 
                          imgsz=640, 
                          batch=4,
                          workers=0,
                          device=0,
                          optimizer="SGD",      # 要使用的优化器，choices=["SGD", "Adam", "AdamW", "RMSProp"]
                          # device=[0, 1]       # 多GPU训练
                          pretrained=True,      # 是否使用预训练模型
                          amp=False,            # 自动混合精度（AMP）训练
                          )


    # 预测 - 预测时应该使用训练的最好的模型
    # model = YOLO("I:\\ultralytics\\runs\\detect\\train6\\weights\\best.pt")
    # metrics = model.val()  # 在验证集上评估模型性能

    # 对图像进行预测
    # img_path1 = "https://ultralytics.com/images/bus.jpg"
    # img_path = "H:\\ultralytics\\detasets\\Axis\\images\\test\\2ST-3_4_A2-4-3_20240110134723.jpg"
    # results = model(img_path)

    # 将模型导出为 ONNX 格式
    # success = model.export(format="onnx") 


if __name__ == '__main__':
    main() 