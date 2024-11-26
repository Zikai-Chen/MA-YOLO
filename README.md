MA-YOLO
===
Enhanced Multi-Scale Attentional Remote Sensing Detector

Usage Guidelines
===
Yolov8 related codes can be obtained from the following links:https://github.com/ultralytics/ultralytics  
Requirements.txt contains the required environment configuration, and the yolo environment can be configured through the following code:pip install -r requirements.txt  
The cfg file contains the YAML file of the dataset, the YAML file of the model, and the experimental parameter settings.  
nn.mymodules contains the innovative modules of the MA-YOLO model.

Detailed Documentation
===
First, download the yolov8 related code and configure the required environment. Then, import the MA-YOLO module in the required location of the model and create the dataset yaml file and model yaml file before running. The following is the model training code:  
  
import warnings  
warnings.filterwarnings('ignore')  
from ultralytics import YOLO  

if __name__ == '__main__':
    model = YOLO(r'E:\PycharmProjects\yolov8.2\ultralytics-8.2.48\ultralytics-8.2.48\ultralytics\cfg\models\DIOR\yolov8n.yaml')
    # 如何切换模型版本, 上面的ymal文件可以改为 yolov8s.yaml就是使用的v8s,
    # 类似某个改进的yaml文件名称为yolov8-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov8l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！

    # model.load('yolov8n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度

    model.train(data=r'E:\PycharmProjects\yolov8.2\ultralytics-8.2.48\ultralytics-8.2.48\ultralytics\cfg\datasets\DIOR.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                imgsz=640,
                epochs=200,
                batch=16,
                workers=8,
                project='runs/train',
                name='exp',
                )

Datasets
===
The two datasets used in this research can be downloaded from [DIOR](https://gitcode.com/Resource-Bundle-Collection/b7f4f/overview),[TGRS-HRRSD](https://github.com/CrazyStoneonRoad/TGRS-HRRSD-Dataset).
