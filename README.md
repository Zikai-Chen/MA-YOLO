MA-YOLO
===
Enhanced Multi-Scale Attentional Remote Sensing Detector

Abstract
===
With the continuous development of deep learning technology, object detection tasks in remote sensing images have received increasing attention. However, due to the diversity of target scales and the complexity of background environments, current detectors often find it difficult to control computational costs while ensuring high performance. To address these challenges, we design a remote sensing image object detector called MA-YOLO, which integrates multi-scale features and attention mechanisms. We introduce the mixed receptive field attention convolution (MRFAConv) technique to strengthen the backbone network and achieve dual improvements in spatial and channel attention through non-shared parameter convolution. In addition a multi-scale receptive field downsampling module (MRFD) is proposed, which can extract rich feature information from different receptive fields while effectively reducing information loss. Ultimately, a lightweight multi-scale attention module (LMSA) is designed and integrated into the neck network to further optimize the feature fusion effect. Extensive experiments on the DIOR and TGRS-HRRSD datasets show that MA-YOLO improves the mean average precision (mAP) by 2.1% and 4% respectively compared to the baseline model YOLOv8n, while maintaining similar computational complexity and reducing the number of parameters by 6.7%. These experimental results fully demonstrate the remarkable effect of our proposed method in improving the detection accuracy of remote sensing images.

Datasets
===
The two datasets used in this research can be downloaded from [DIOR](https://gitcode.com/Resource-Bundle-Collection/b7f4f/overview),[TGRS-HRRSD](https://github.com/CrazyStoneonRoad/TGRS-HRRSD-Dataset).
