MA-YOLO
===
Enhanced Multi-Scale Attentional Remote Sensing Detector

Usage Guideline
===
YOLOv8 related codes can be obtained from the following links: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/tree/v8.2.48)  
Requirements.txt contains the required environment configuration, and the yolo environment can be configured through the following code: pip install -r requirements.txt  
The cfg file contains the yaml file of the dataset, the yaml file of the model, and the experimental parameter settings.  
nn.mymodules contains the innovative modules of the MA-YOLO model.

Detailed Documentation
===
First, download the YOLOv8 related code and configure the required environment. Then, import the MA-YOLO module in the required location of the model and create the dataset yaml file and model yaml file before running. The following is the model training code:  

```
import warnings  
warnings.filterwarnings('ignore')  
from ultralytics import YOLO  

if __name__ == '__main__':
    model = YOLO(r'E:\PycharmProjects\yolov8.2\ultralytics-8.2.48\ultralytics-8.2.48\ultralytics\cfg\models\DIOR\yolov8n.yaml')
    # model.load('yolov8n.pt')

    model.train(data=r'E:\PycharmProjects\yolov8.2\ultralytics-8.2.48\ultralytics-8.2.48\ultralytics\cfg\datasets\DIOR.yaml',
                imgsz=640,
                epochs=200,
                batch=16,
                workers=8,
                project='runs/train',
                name='exp',
                )
```

Datasets
===
The two datasets used in this research can be downloaded from [DIOR](https://gitcode.com/Resource-Bundle-Collection/b7f4f/overview),[TGRS-HRRSD](https://github.com/CrazyStoneonRoad/TGRS-HRRSD-Dataset).
