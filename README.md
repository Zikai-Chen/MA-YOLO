MA-YOLO
===
With the continuous development of deep learning technology, object detection tasks in remote sensing images have received increasing attention. However, due to the diversity of object scales and the complexity of background environments, current detectors often find it difficult to control computational costs while ensuring high performance. To address these challenges, we design a remote sensing image object detector called MA-YOLO, which integrates multi-scale features and attention mechanisms. We design the mixed receptive field attention convolution (MRFAConv) module to strengthen the backbone network, which is a non-parametric shared convolution that takes into account both spatial and channel attention. Moreover, a multi-scale receptive field downsampling module (MRFD) is proposed, which can extract rich feature information from different receptive fields while effectively reducing information loss. Ultimately, a lightweight multi-scale attention module (LMSA) is designed and integrated into the neck network to further optimize the feature fusion effect. Extensive experiments on the DIOR and the TGRS-HRRSD datasets show that MA-YOLO improves the mAP by 2.1% and 4% respectively compared to the baseline model YOLOv8n, while maintaining similar computational overhead and reducing the number of parameters by 6.7%. These experimental results fully demonstrate the remarkable effect of our proposed method in improving the detection accuracy of remote sensing images.

Usage Guideline
===
The related source codes of YOLOv8 can be obtained from the following links: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/tree/v8.2.48)  
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
